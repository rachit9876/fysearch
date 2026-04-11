from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pypdf import PdfReader

from .db import Document, get_extracted_text, upsert_document, upsert_extracted_text
from .config import load_config
from .paths import get_paths


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def extract_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    parts: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            parts.append(text)
    return "\n\n".join(parts).strip()


def extract_pdf_text_pages(pdf_path: Path) -> list[str]:
    """Return selectable text per page (empty strings allowed)."""
    reader = PdfReader(str(pdf_path))
    out: list[str] = []
    for page in reader.pages:
        text = (page.extract_text() or "").strip()
        out.append(text)
    return out


def _get_ocr_languages() -> str:
    try:
        cfg = load_config()
        langs = (getattr(cfg, "ocr_languages", "") or "").strip()
        return langs or "eng"
    except Exception:
        return "eng"


def ocr_image(image_path: Path, *, languages: Optional[str] = None) -> tuple[str, Optional[float], Optional[str]]:
    # Optional dependency: pytesseract (+ system tesseract)
    try:
        import pytesseract  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("pytesseract not installed; install extras: pip install -e .[ocr]") from e

    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    langs = (languages or "").strip() or _get_ocr_languages()
    text = pytesseract.image_to_string(img, lang=langs)
    return text.strip(), None, None


def _ocr_single_page(args: tuple[str, Path]) -> tuple[str, str, Optional[float], Optional[str]]:
    """Worker function for parallel OCR. Returns (page_doc_id, text, conf, lang)."""
    page_doc_id, page_img_path = args
    try:
        text, conf, lang = ocr_image(page_img_path)
        return (page_doc_id, text, conf, lang)
    except Exception:
        return (page_doc_id, "", None, None)


def _derive_page_doc_id(pdf_doc_id: str, page_number: int) -> str:
    # Deterministic id so we don't duplicate work across runs.
    h = hashlib.sha256()
    h.update(f"{pdf_doc_id}:page:{page_number}".encode("utf-8"))
    return h.hexdigest()


def _pdf_pages_already_ingested(conn, original_pdf_path: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM documents WHERE original_path LIKE ? LIMIT 1",
        (f"{original_pdf_path}#page=%",),
    ).fetchone()
    return row is not None


def _render_pdf_pages_to_store(
    *,
    conn,
    pdf_doc_id: str,
    stored_pdf_path: Path,
    original_pdf_path: str,
    dpi: int = 200,
    max_threads: Optional[int] = None,
) -> list[tuple[str, Path]]:
    """Render a PDF into per-page JPEGs under data/store and upsert them as image documents.

    Uses pdf2image + poppler when available. If not installed, returns [].
    """
    try:
        from pdf2image import convert_from_path  # type: ignore
    except Exception:
        return []

    paths = get_paths()
    paths.store_dir.mkdir(parents=True, exist_ok=True)

    # Use config effective_max_workers if not specified
    if max_threads is None:
        cfg = load_config()
        max_threads = cfg.effective_max_workers
    threads = max(1, min(max_threads, (os.cpu_count() or 1)))
    rendered: list[tuple[str, Path]] = []

    with tempfile.TemporaryDirectory(prefix="fysearch_pdf_") as tmpdir:
        page_paths = convert_from_path(
            str(stored_pdf_path),
            dpi=dpi,
            fmt="jpeg",
            output_folder=tmpdir,
            paths_only=True,
            thread_count=threads,
        )

        for page_number, tmp_path in enumerate(page_paths, start=1):
            page_doc_id = _derive_page_doc_id(pdf_doc_id, page_number)
            out_path = (paths.store_dir / f"{page_doc_id}.jpg").resolve()
            if not out_path.exists():
                shutil.copy2(tmp_path, out_path)

            # Store as an image document so it can be embedded/indexed.
            upsert_document(
                conn,
                Document(
                    doc_id=page_doc_id,
                    original_path=f"{original_pdf_path}#page={page_number}",
                    stored_path=str(out_path),
                    media_type="image",
                    created_at=_now_iso(),
                ),
            )
            rendered.append((page_doc_id, out_path))

    return rendered


def extract_text_for_doc(conn, doc_id: str, stored_path: Path, media_type: str) -> bool:
    existing = get_extracted_text(conn, doc_id)
    existing_text = (existing["text"] if existing is not None else "")
    has_existing_text = bool((existing_text or "").strip())

    if media_type == "pdf":
        # We always try to create per-page image docs (CPU-friendly DPI) so PDFs
        # can participate in image indexing/search.
        row = conn.execute("SELECT original_path FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        original_pdf_path = str(row["original_path"]) if row else str(stored_path)

        pages_present = _pdf_pages_already_ingested(conn, original_pdf_path)
        rendered_pages: list[tuple[str, Path]] = []
        if not pages_present:
            rendered_pages = _render_pdf_pages_to_store(
                conn=conn,
                pdf_doc_id=doc_id,
                stored_pdf_path=stored_path,
                original_pdf_path=original_pdf_path,
            )

        if pages_present and not rendered_pages:
            page_rows = conn.execute(
                "SELECT doc_id, stored_path FROM documents WHERE original_path LIKE ? ORDER BY original_path",
                (f"{original_pdf_path}#page=%",),
            ).fetchall()
            rendered_pages = [(str(r["doc_id"]), Path(str(r["stored_path"]))) for r in page_rows]

        changed = False

        page_texts: list[str] = []
        any_selectable = False
        try:
            page_texts = extract_pdf_text_pages(stored_path)
            any_selectable = any(t.strip() for t in page_texts)
        except Exception:
            page_texts = []
            any_selectable = False

        # Always backfill per-page extracted text when possible, even if the
        # PDF doc already has extracted text. This ensures text search can land
        # on the correct page doc (and avoids broken thumbnails for PDF hits).
        if rendered_pages and page_texts:
            for i, text in enumerate(page_texts, start=1):
                if not text.strip():
                    continue
                page_doc_id = _derive_page_doc_id(doc_id, i)
                page_existing = get_extracted_text(conn, page_doc_id)
                page_has_text = bool(str(page_existing["text"] or "").strip()) if page_existing is not None else False
                if not page_has_text:
                    upsert_extracted_text(
                        conn,
                        page_doc_id,
                        text,
                        method="pdf_text",
                        confidence=None,
                        language=None,
                        created_at=_now_iso(),
                    )
                    changed = True

        # If the PDF itself has no extracted text yet, store a combined record.
        if not has_existing_text:
            if any_selectable and page_texts:
                combined = "\n\n".join([t for t in page_texts if t.strip()]).strip()
                upsert_extracted_text(conn, doc_id, combined, method="pdf_text", confidence=None, language=None, created_at=_now_iso())
                changed = True
            else:
                # If no selectable text, attempt OCR on missing per-page docs in parallel.
                ocr_parts: list[str] = []
                pages_to_ocr: list[tuple[str, Path]] = []
                
                for page_doc_id, page_img_path in rendered_pages:
                    page_existing = get_extracted_text(conn, page_doc_id)
                    page_has_text = bool(str(page_existing["text"] or "").strip()) if page_existing is not None else False
                    if page_existing is None or not page_has_text:
                        pages_to_ocr.append((page_doc_id, page_img_path))
                
                # Parallel OCR processing - use all logical processors
                if pages_to_ocr:
                    cfg = load_config()
                    max_w = cfg.effective_max_workers  # Will be 8 on your system
                    with ThreadPoolExecutor(max_workers=max_w) as executor:
                        futures = [executor.submit(_ocr_single_page, args) for args in pages_to_ocr]
                        for future in as_completed(futures):
                            page_doc_id, page_text, conf, lang = future.result()
                            upsert_extracted_text(conn, page_doc_id, page_text, method="ocr", confidence=conf, language=lang, created_at=_now_iso())
                            changed = True
                
                # Collect all OCR'd text
                for page_doc_id, _ in rendered_pages:
                    page_row = get_extracted_text(conn, page_doc_id)
                    if page_row is not None:
                        t = str(page_row["text"] or "").strip()
                        if t:
                            ocr_parts.append(t)

                combined = "\n\n".join(ocr_parts).strip()
                upsert_extracted_text(
                    conn,
                    doc_id,
                    combined,
                    method="ocr",
                    confidence=None,
                    language=None,
                    created_at=_now_iso(),
                )
                changed = True

        return changed or (not pages_present and bool(rendered_pages))

    if media_type == "image":
        # OCR is optional; we store empty text if OCR isn't configured.
        if existing is not None and has_existing_text:
            return False
        try:
            text, conf, lang = ocr_image(stored_path)
        except Exception:
            text, conf, lang = "", None, None
        upsert_extracted_text(conn, doc_id, text, method="ocr", confidence=conf, language=lang, created_at=_now_iso())
        return True

    if media_type == "text":
        if existing is not None and has_existing_text:
            return False
        try:
            text = stored_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            text = ""
        upsert_extracted_text(conn, doc_id, text, method="plain_text", confidence=1.0, language=None, created_at=_now_iso())
        return True

    return False
