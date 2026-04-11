from __future__ import annotations

import hashlib
import mimetypes
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .config import load_config
from .db import Document, upsert_document
from .paths import get_paths


@dataclass(frozen=True)
class IngestResult:
    doc_id: str
    stored_path: Path
    media_type: str
    is_new: bool  # True if file was newly added, False if already existed


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _detect_media_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return "pdf"
    if ext == ".txt":
        return "text"
    if ext in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}:
        return "image"
    mime, _ = mimetypes.guess_type(str(path))
    if mime and mime.startswith("image/"):
        return "image"
    return "unknown"


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    # Use 8MB read buffer for faster I/O on large files
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _process_single_file(file_path: Path, store_dir: Path) -> Optional[tuple[str, Path, str, Path, bool]]:
    """Process a single file: hash, detect type, copy if needed. 
    Returns (doc_id, stored_path, media_type, original_path, is_new) or None on error."""
    try:
        doc_id = _hash_file(file_path)
        media_type = _detect_media_type(file_path)
        stored_name = f"{doc_id}{file_path.suffix.lower()}"
        stored_path = (store_dir / stored_name).resolve()
        is_new = not stored_path.exists()
        if is_new:
            shutil.copy2(file_path, stored_path)
        return (doc_id, stored_path, media_type, file_path, is_new)
    except Exception:
        return None


def ingest_path(conn, src: Path, max_workers: Optional[int] = None) -> list[IngestResult]:
    paths = get_paths()
    src = src.resolve()
    paths.store_dir.mkdir(parents=True, exist_ok=True)

    # Determine worker count from config
    if max_workers is None:
        cfg = load_config()
        max_workers = cfg.effective_max_workers

    candidates: list[Path]
    if src.is_dir():
        candidates = [p for p in src.rglob("*") if p.is_file()]
    else:
        candidates = [src]

    results: list[IngestResult] = []
    processed_files: list[tuple[str, Path, str, Path, bool]] = []

    # ThreadPoolExecutor is better for I/O-bound work (file hashing + copy).
    # Avoids the heavy process spawn overhead of ProcessPoolExecutor.
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_single_file, fp, paths.store_dir): fp for fp in candidates}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                processed_files.append(result)

    # Sequential DB upserts (SQLite is not thread-safe for writes)
    for doc_id, stored_path, media_type, original_path, is_new in processed_files:
        upsert_document(
            conn,
            Document(
                doc_id=doc_id,
                original_path=str(original_path),
                stored_path=str(stored_path),
                media_type=media_type,
                created_at=_now_iso(),
            ),
        )
        results.append(IngestResult(doc_id=doc_id, stored_path=stored_path, media_type=media_type, is_new=is_new))

    return results
