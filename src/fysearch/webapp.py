from __future__ import annotations

import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import shutil

import numpy as np

from .config import load_config
from .db import add_search_history, clear_search_history, connect, init_db, list_documents, list_search_history
from .embeddings import maybe_image_embedder, maybe_text_embedder
from .extract import extract_text_for_doc
from .ingest import ingest_path
from .paths import get_paths, normalize_path
from .vector_index import BruteForceIndex, FaissIndex


@dataclass(frozen=True)
class ResultRow:
    score: float
    doc_id: str
    stored_path: str
    original_path: str
    media_type: str
    method: str
    snippet: str
    page: str


_RUNTIME_CACHE_LOCK = threading.RLock()
_TEXT_EMBEDDER_CACHE: dict[str, Any] = {}
_IMAGE_EMBEDDER_CACHE: dict[str, Any] = {}
_INDEX_CACHE: dict[tuple[str, bool], dict[str, Any]] = {}


def _index_path_for_modality(modality: str) -> Path:
    paths = get_paths()
    name = "text_index.npz" if modality == "text" else "image_index.npz"
    return paths.index_dir / name


def _index_file_stamp(npz: Path) -> tuple[int, int]:
    st = npz.stat()
    return (int(st.st_mtime_ns), int(st.st_size))


def _clear_runtime_caches() -> None:
    with _RUNTIME_CACHE_LOCK:
        _TEXT_EMBEDDER_CACHE.clear()
        _IMAGE_EMBEDDER_CACHE.clear()
        _INDEX_CACHE.clear()


def _get_cached_text_embedder(model_name_or_path: str):
    if not model_name_or_path:
        raise RuntimeError("Config.text_model is empty")

    with _RUNTIME_CACHE_LOCK:
        cached = _TEXT_EMBEDDER_CACHE.get(model_name_or_path)
    if cached is not None:
        return cached

    embedder = maybe_text_embedder(model_name_or_path)
    if embedder is None:
        raise RuntimeError("Config.text_model is empty")

    with _RUNTIME_CACHE_LOCK:
        _TEXT_EMBEDDER_CACHE[model_name_or_path] = embedder
    return embedder


def _get_cached_image_embedder(model_name_or_path: str):
    if not model_name_or_path:
        raise RuntimeError("Config.image_model is empty")

    with _RUNTIME_CACHE_LOCK:
        cached = _IMAGE_EMBEDDER_CACHE.get(model_name_or_path)
    if cached is not None:
        return cached

    embedder = maybe_image_embedder(model_name_or_path)
    if embedder is None:
        raise RuntimeError("Config.image_model is empty")

    with _RUNTIME_CACHE_LOCK:
        _IMAGE_EMBEDDER_CACHE[model_name_or_path] = embedder
    return embedder


def _page_from_original_path(original_path: str) -> str:
    marker = "#page="
    if marker not in original_path:
        return ""
    try:
        return original_path.split(marker, 1)[1].strip()
    except Exception:
        return ""


def _make_snippet(text: str, query: str, max_len: int = 220) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    q = (query or "").strip().lower()
    if not q:
        return (text[:max_len] + ("…" if len(text) > max_len else "")).strip()

    tokens = [t for t in q.replace("\n", " ").split() if len(t) >= 3]
    hay = text.lower()
    idx = -1
    for tok in tokens[:8]:
        idx = hay.find(tok)
        if idx != -1:
            break
    if idx == -1:
        return (text[:max_len] + ("…" if len(text) > max_len else "")).strip()

    start = max(0, idx - max_len // 2)
    end = min(len(text), start + max_len)
    snippet = text[start:end].strip()
    if start > 0:
        snippet = "…" + snippet
    if end < len(text):
        snippet = snippet + "…"
    return snippet


def _get_index(dim: int, prefer_faiss: bool):
    if prefer_faiss:
        try:
            return FaissIndex(dim)
        except Exception:
            return BruteForceIndex(dim)
    return BruteForceIndex(dim)


def _get_cached_index(modality: str, prefer_faiss: bool):
    npz = _index_path_for_modality(modality)
    if not npz.exists():
        raise FileNotFoundError(f"Missing index: {npz}")

    stamp = _index_file_stamp(npz)
    key = (modality, prefer_faiss)

    with _RUNTIME_CACHE_LOCK:
        cached = _INDEX_CACHE.get(key)
        if cached is not None and cached.get("stamp") == stamp:
            return (cached["doc_ids"], cached["index"])

    data = np.load(npz, allow_pickle=True)
    doc_ids = [str(x) for x in data["doc_ids"]]
    vectors = data["vectors"].astype(np.float32)

    dim = vectors.shape[1]
    idx = _get_index(dim=dim, prefer_faiss=prefer_faiss)
    idx.add(doc_ids, vectors)

    with _RUNTIME_CACHE_LOCK:
        _INDEX_CACHE[key] = {
            "stamp": stamp,
            "doc_ids": doc_ids,
            "index": idx,
        }
    return (doc_ids, idx)


def _build_text_index(prefer_faiss: bool) -> Path:
    cfg = load_config()
    embedder = _get_cached_text_embedder(cfg.text_model)

    conn = connect()
    init_db(conn)
    rows = conn.execute(
        """
        SELECT d.doc_id, t.text
        FROM documents d
        JOIN extracted_text t ON t.doc_id = d.doc_id
        WHERE t.text IS NOT NULL
          AND length(trim(t.text)) > 0
          AND (
            d.original_path LIKE '%#page=%'
            OR d.media_type NOT IN ('pdf', 'image')
            OR (
                d.media_type = 'pdf'
                AND NOT EXISTS (
                    SELECT 1
                    FROM documents p
                    WHERE p.original_path LIKE d.original_path || '#page=%'
                )
            )
          )
        """
    ).fetchall()
    conn.close()

    doc_ids: list[str] = []
    texts: list[str] = []
    for r in rows:
        text = (r["text"] or "").strip()
        if text:
            doc_ids.append(r["doc_id"])
            texts.append(text)

    if not texts:
        raise RuntimeError("No extracted text available to index. Enable OCR or ingest PDFs with selectable text.")

    mat = embedder.embed_batch(texts, batch_size=256)

    paths = get_paths()
    out = paths.index_dir / "text_index.npz"
    np.savez_compressed(out, doc_ids=np.asarray(doc_ids), vectors=mat)
    return out


def _build_image_index(prefer_faiss: bool) -> Path:
    cfg = load_config()
    embedder = _get_cached_image_embedder(cfg.image_model)

    conn = connect()
    init_db(conn)
    rows = conn.execute(
        """
        SELECT doc_id, stored_path
        FROM documents
        WHERE media_type = 'image'
        """
    ).fetchall()
    conn.close()

    if not rows:
        raise RuntimeError("No images found to index.")

    doc_ids = [r["doc_id"] for r in rows]
    image_paths = [r["stored_path"] for r in rows]

    mat = embedder.embed_batch(image_paths, batch_size=128)

    paths = get_paths()
    out = paths.index_dir / "image_index.npz"
    np.savez_compressed(out, doc_ids=np.asarray(doc_ids), vectors=mat)
    return out


def text_query(query: str, target_modality: str, top_k: int, prefer_faiss: bool) -> list[ResultRow]:
    cfg = load_config()
    text_embedder = _get_cached_text_embedder(cfg.text_model)

    q = text_embedder.embed(query).astype(np.float32)
    return _text_query_from_vector(
        query=query,
        query_vec=q,
        target_modality=target_modality,
        top_k=top_k,
        prefer_faiss=prefer_faiss,
    )


def _text_query_from_vector(
    *,
    query: str,
    query_vec: np.ndarray,
    target_modality: str,
    top_k: int,
    prefer_faiss: bool,
) -> list[ResultRow]:
    _, idx = _get_cached_index(target_modality, prefer_faiss=prefer_faiss)

    hits = idx.search(query_vec, top_k)

    conn = connect()
    init_db(conn)

    results: list[ResultRow] = []
    for hit in hits:
        row = conn.execute(
            """
            SELECT d.stored_path, d.original_path, d.media_type, t.method, t.text
            FROM documents d
            LEFT JOIN extracted_text t ON t.doc_id = d.doc_id
            WHERE d.doc_id = ?
            """,
            (hit.doc_id,),
        ).fetchone()
        if not row:
            continue

        original_path = str(row["original_path"])
        text = str(row["text"] or "")

        results.append(ResultRow(
            score=hit.score,
            doc_id=hit.doc_id,
            stored_path=str(row["stored_path"]),
            original_path=original_path,
            media_type=str(row["media_type"]),
            method=str(row["method"] or ""),
            snippet=_make_snippet(text, query=query),
            page=_page_from_original_path(original_path),
        ))

    conn.close()
    return results


def auto_query(query: str, top_k: int, prefer_faiss: bool) -> tuple[list[ResultRow], list[ResultRow]]:
    """Smart auto search: returns separate text and image results."""
    cfg = load_config()
    text_embedder = _get_cached_text_embedder(cfg.text_model)
    query_vec = text_embedder.embed(query).astype(np.float32)

    text_results: list[ResultRow] = []
    image_results: list[ResultRow] = []

    try:
        text_results = _text_query_from_vector(
            query=query,
            query_vec=query_vec,
            target_modality="text",
            top_k=top_k,
            prefer_faiss=prefer_faiss,
        )
    except (FileNotFoundError, RuntimeError):
        pass

    try:
        image_results = _text_query_from_vector(
            query=query,
            query_vec=query_vec,
            target_modality="image",
            top_k=top_k,
            prefer_faiss=prefer_faiss,
        )
    except (FileNotFoundError, RuntimeError):
        pass

    return (text_results, image_results)


def image_query(image_path: Path, top_k: int, prefer_faiss: bool) -> list[ResultRow]:
    cfg = load_config()
    image_embedder = _get_cached_image_embedder(cfg.image_model)

    _, idx = _get_cached_index("image", prefer_faiss=prefer_faiss)

    q = image_embedder.embed(str(image_path)).astype(np.float32)
    hits = idx.search(q, top_k)

    conn = connect()
    init_db(conn)

    out: list[ResultRow] = []
    for hit in hits:
        row = conn.execute(
            "SELECT stored_path, original_path, media_type FROM documents WHERE doc_id = ?",
            (hit.doc_id,),
        ).fetchone()
        if not row:
            continue
        original_path = str(row["original_path"])
        out.append(
            ResultRow(
                score=hit.score,
                doc_id=hit.doc_id,
                stored_path=str(row["stored_path"]),
                original_path=original_path,
                media_type=str(row["media_type"]),
                method="",
                snippet="",
                page=_page_from_original_path(original_path),
            )
        )

    conn.close()
    return out


def create_app():
    try:
        from flask import Flask, abort, flash, get_flashed_messages, redirect, render_template, request, send_file, url_for
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Flask not installed. Run: pip install -e '.[web]'") from e

    app = Flask(__name__)
    app.secret_key = 'fysearch-dev-key-change-in-production'

    # ── Helpers ──────────────────────────────────────────────────────────

    def _last_uploaded_filename_from_history(history: list[dict[str, str]]) -> Optional[str]:
        for h in history:
            if (h.get("kind") or "").lower() != "image":
                continue
            p = (h.get("uploaded_path") or "").strip()
            if not p:
                continue
            name = Path(p).name
            if not name or Path(name).name != name:
                continue
            return name
        return None

    def _index_status() -> dict[str, bool]:
        paths = get_paths()
        return {
            "has_text_index": (paths.index_dir / "text_index.npz").exists(),
            "has_image_index": (paths.index_dir / "image_index.npz").exists(),
        }

    def _get_history() -> list[dict[str, str]]:
        conn = connect()
        init_db(conn)
        rows = list_search_history(conn, limit=25)
        conn.close()
        out: list[dict[str, str]] = []
        for r in rows:
            out.append(
                {
                    "created_at": str(r["created_at"]),
                    "kind": str(r["kind"]),
                    "query_text": str(r["query_text"] or ""),
                    "target_modality": str(r["target_modality"] or ""),
                    "top_k": str(r["top_k"] or ""),
                    "uploaded_path": str(r["uploaded_path"] or ""),
                }
            )
        return out

    def _render(
        *,
        text_results=None,
        image_results=None,
        error=None,
        message=None,
        query="",
        top_k=1,
        modality="auto",
        search_kind="",
        last_uploaded_filename=None,
    ):
        """Single render helper — eliminates 6+ duplicate render_template calls."""
        cfg = load_config()
        history = _get_history()
        return render_template(
            "index.html",
            text_results=text_results or None,
            image_results=image_results or None,
            error=error,
            message=message,
            query=query,
            top_k=top_k,
            modality=modality,
            search_kind=search_kind,
            dataset_path=cfg.dataset_path,
            history=history,
            last_uploaded_filename=last_uploaded_filename,
            **_index_status(),
        )

    # ── Routes ───────────────────────────────────────────────────────────

    @app.get("/upload/<name>")
    def serve_upload(name: str):
        if not name or Path(name).name != name:
            abort(404)
        paths = get_paths()
        uploads_root = (paths.data_dir / "uploads").resolve()
        fp = (uploads_root / name).resolve()
        if uploads_root not in fp.parents or not fp.exists() or not fp.is_file():
            abort(404)
        return send_file(fp)

    def _reset_all_data() -> None:
        paths = get_paths()

        cfg = load_config()
        if getattr(cfg, "dataset_path", ""):
            cfg.dataset_path = ""
            from .config import save_config
            save_config(cfg)

        for folder in [paths.store_dir, paths.index_dir, paths.data_dir / "uploads"]:
            if folder.exists():
                shutil.rmtree(folder, ignore_errors=True)

        if paths.db_path.exists():
            try:
                paths.db_path.unlink()
            except Exception:
                pass

        paths.store_dir.mkdir(parents=True, exist_ok=True)
        paths.index_dir.mkdir(parents=True, exist_ok=True)
        (paths.data_dir / "uploads").mkdir(parents=True, exist_ok=True)
        paths.db_dir.mkdir(parents=True, exist_ok=True)

        conn = connect()
        init_db(conn)
        conn.close()
        _clear_runtime_caches()

    @app.get("/")
    def index():
        cfg = load_config()
        history = _get_history()
        flashes = get_flashed_messages(with_categories=True)
        error = None
        message = None
        for category, msg in flashes:
            if category == 'error':
                error = msg
            else:
                message = msg
        return render_template(
            "index.html",
            text_results=None,
            image_results=None,
            error=error,
            message=message,
            query="",
            top_k=1,
            modality="auto",
            search_kind="",
            dataset_path=cfg.dataset_path,
            history=history,
            last_uploaded_filename=None,
            **_index_status(),
        )

    @app.post("/history/clear")
    def clear_history():
        conn = connect()
        init_db(conn)
        uploaded_paths = clear_search_history(conn)
        conn.close()

        paths = get_paths()
        uploads_root = (paths.data_dir / "uploads").resolve()
        deleted = 0
        for p in uploaded_paths:
            try:
                fp = Path(p).resolve()
                if uploads_root in fp.parents and fp.exists() and fp.is_file():
                    fp.unlink()
                    deleted += 1
            except Exception:
                continue

        flash(f"Cleared history. Deleted {deleted} uploaded query images.", "success")
        return redirect(url_for("index"))

    @app.post("/reset")
    def reset():
        confirm = (request.form.get("confirm") or "").strip().lower() == "on"
        if not confirm:
            flash("Reset not confirmed. Tick the checkbox and try again.", "error")
            return redirect(url_for("index"))

        _reset_all_data()
        flash("Reset complete. Local store/index/DB/uploads cleared.", "success")
        return redirect(url_for("index"))

    @app.post("/build-index")
    def build_index_route():
        modality = (request.form.get("modality") or "image").strip().lower()
        prefer_faiss = True

        if modality not in {"text", "image"}:
            flash("Invalid modality. Choose text or image.", "error")
            return redirect(url_for("index"))

        try:
            if modality == "image":
                _build_image_index(prefer_faiss=prefer_faiss)
                _clear_runtime_caches()
                flash("Image index built successfully!", "success")
            else:
                _build_text_index(prefer_faiss=prefer_faiss)
                _clear_runtime_caches()
                flash("Text index built successfully!", "success")
        except Exception as e:
            flash(f"Failed to build {modality} index: {str(e)}", "error")

        return redirect(url_for("index"))

    @app.post("/dataset")
    def set_dataset():
        cfg = load_config()
        dataset_path = (request.form.get("dataset_path") or "").strip()
        run_pipeline = (request.form.get("run_pipeline") or "").strip().lower() == "on"
        prefer_faiss = True

        if not dataset_path:
            flash("Dataset folder path is empty", "error")
            return redirect(url_for("index"))

        try:
            dataset_path = normalize_path(dataset_path)
            p = Path(dataset_path).expanduser()

            if not p.exists():
                flash(f"Folder does not exist: {dataset_path}", "error")
                return redirect(url_for("index"))
            if not p.is_dir():
                flash(f"Path is not a directory: {dataset_path}", "error")
                return redirect(url_for("index"))

            p = p.resolve()
        except Exception as e:
            flash(f"Invalid path: {dataset_path} - {str(e)}", "error")
            return redirect(url_for("index"))

        cfg.dataset_path = str(p)
        from .config import save_config
        save_config(cfg)

        if run_pipeline:
            try:
                conn = connect()
                init_db(conn)
                ingest_results = ingest_path(conn, p)
                new_files = sum(1 for r in ingest_results if r.is_new)
                existing_files = len(ingest_results) - new_files
                extracted = 0
                for row in list_documents(conn):
                    if extract_text_for_doc(conn, row["doc_id"], Path(row["stored_path"]), row["media_type"]):
                        extracted += 1
                conn.close()

                built = []
                errors = []
                try:
                    _build_image_index(prefer_faiss=prefer_faiss)
                    built.append("image")
                except Exception as e:
                    errors.append(f"Image index failed: {str(e)}")
                try:
                    _build_text_index(prefer_faiss=prefer_faiss)
                    built.append("text")
                except Exception as e:
                    errors.append(f"Text index failed: {str(e)}")

                if built:
                    _clear_runtime_caches()

                msg = f"Scanned {len(ingest_results)} files ({new_files} new, {existing_files} already indexed), extracted {extracted} docs, built: {', '.join(built) or 'none'}"
                if errors:
                    msg += f" | Errors: {'; '.join(errors)}"
                flash(msg, "success" if built else "error")
            except Exception as e:
                flash(str(e), "error")
        else:
            flash("Dataset path saved.", "success")

        return redirect(url_for("index"))

    @app.post("/search/text")
    def search_text_post():
        query = (request.form.get("query") or "").strip()
        modality = (request.form.get("modality") or "auto").strip().lower()
        top_k = int(request.form.get("top_k") or 1)

        if not query:
            return redirect(url_for("index"))

        conn = connect()
        add_search_history(
            conn,
            kind="text",
            query_text=query,
            target_modality=modality,
            top_k=top_k,
            uploaded_path=None,
        )
        conn.close()

        return redirect(url_for("search_text_get", q=query, modality=modality, top_k=top_k))

    @app.get("/search/text")
    def search_text_get():
        query = (request.args.get("q") or "").strip()
        modality = (request.args.get("modality") or "auto").strip().lower()
        top_k = int(request.args.get("top_k") or 1)
        prefer_faiss = True

        if not query:
            return _render(top_k=top_k, modality=modality, search_kind="text")

        try:
            if modality == "auto":
                text_results, image_results = auto_query(query=query, top_k=top_k, prefer_faiss=prefer_faiss)
            else:
                text_results = text_query(query=query, target_modality=modality, top_k=top_k, prefer_faiss=prefer_faiss)
                image_results = None
        except FileNotFoundError:
            if modality == "text":
                msg = "Text index is missing. Build it from the sidebar or run: fysearch build-index --modality text"
            elif modality == "auto":
                msg = "No indexes found. Build indexes from the sidebar."
            else:
                msg = "Image index is missing. Build it from the sidebar or run: fysearch build-index --modality image"
            return _render(error=msg, query=query, top_k=top_k, modality=modality, search_kind="text")
        except Exception as e:
            return _render(error=str(e), query=query, top_k=top_k, modality=modality, search_kind="text")

        return _render(
            text_results=text_results,
            image_results=image_results,
            query=query,
            top_k=top_k,
            modality=modality,
            search_kind="text",
        )

    @app.post("/search/image")
    def search_image():
        top_k = int(request.form.get("top_k") or 1)
        prefer_faiss = True

        file = request.files.get("image")
        if file is None or file.filename == "":
            return _render(error="No image uploaded", top_k=top_k, modality="image", search_kind="image")

        paths = get_paths()
        upload_dir = paths.data_dir / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)

        suffix = Path(file.filename).suffix or ".img"
        with tempfile.NamedTemporaryFile(delete=False, dir=upload_dir, suffix=suffix) as tmp:
            file.save(tmp)
            tmp_path = Path(tmp.name)

        uploaded_filename = tmp_path.name

        try:
            results = image_query(image_path=tmp_path, top_k=top_k, prefer_faiss=prefer_faiss)
        except FileNotFoundError:
            return _render(
                error="Image index is missing. Build it from the sidebar.",
                top_k=top_k,
                modality="image",
                search_kind="image",
                last_uploaded_filename=uploaded_filename,
            )
        except Exception as e:
            return _render(
                error=str(e),
                top_k=top_k,
                modality="image",
                search_kind="image",
                last_uploaded_filename=uploaded_filename,
            )

        conn = connect()
        add_search_history(
            conn,
            kind="image",
            query_text=None,
            target_modality=None,
            top_k=top_k,
            uploaded_path=str(tmp_path),
        )
        conn.close()

        return _render(
            image_results=results,
            top_k=top_k,
            modality="image",
            search_kind="image",
            last_uploaded_filename=uploaded_filename,
        )

    @app.get("/doc/<doc_id>")
    def serve_doc(doc_id: str):
        conn = connect()
        init_db(conn)
        row = conn.execute(
            "SELECT stored_path FROM documents WHERE doc_id = ?",
            (doc_id,),
        ).fetchone()
        conn.close()
        if not row:
            abort(404)
        path = Path(row["stored_path"])
        if not path.exists():
            abort(404)
        return send_file(path)

    return app
