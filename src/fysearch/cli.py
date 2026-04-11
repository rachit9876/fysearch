from __future__ import annotations

import multiprocessing
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from .config import Config, config_to_table, load_config, save_config
from .db import connect, init_db, list_documents
from .embeddings import maybe_image_embedder, maybe_text_embedder
from .extract import extract_text_for_doc
from .ingest import ingest_path
from .paths import get_paths
from .vector_index import BruteForceIndex, FaissIndex

# ---------------------------------------------------------------------------
# Ensure proper multiprocessing start method on Linux/WSL.
# "fork" is fastest and avoids re-importing heavy modules in each worker.
# On macOS 3.13+, default changed to "spawn" which is slower and can
# cause pickling issues with module-level state.
# ---------------------------------------------------------------------------
try:
    if sys.platform != "win32":
        multiprocessing.set_start_method("fork", force=False)
except RuntimeError:
    pass  # Already set — ignore

app = typer.Typer(add_completion=False)
console = Console()


def _extract_worker(row_tuple: tuple) -> bool:
    """Helper for parallel extraction. Each worker needs its own DB connection."""
    doc_id, stored_path_str, media_type, db_path_str = row_tuple
    stored_path = Path(stored_path_str)
    db_path = Path(db_path_str)

    # Open a fresh connection for this worker
    conn = connect(db_path)
    try:
        return extract_text_for_doc(conn, doc_id, stored_path, media_type)
    finally:
        conn.close()


@app.command()
def init() -> None:
    """Initialize local folders + sqlite DB."""
    paths = get_paths()
    paths.input_dir.mkdir(parents=True, exist_ok=True)
    paths.store_dir.mkdir(parents=True, exist_ok=True)
    paths.index_dir.mkdir(parents=True, exist_ok=True)
    paths.db_dir.mkdir(parents=True, exist_ok=True)

    conn = connect(paths.db_path)
    init_db(conn)
    conn.close()

    if not paths.config_path.exists():
        save_config(Config())

    console.print(f"Initialized at: {paths.root}")


@app.command()
def config(
    show: bool = typer.Option(True, help="Show current config"),
    text_model: str = typer.Option("", help="Text model name or local path"),
    image_model: str = typer.Option("", help="Image model name or local path"),
    dataset_path: str = typer.Option("", help="Dataset folder path (optional)"),
    ocr_languages: str = typer.Option("", help="Tesseract OCR languages, e.g. eng, hin, eng+hin"),
    embedding_dim: int = typer.Option(0, help="Embedding dimension"),
    max_workers: int = typer.Option(0, help="Max parallel workers (0 = auto-detect from CPU count)"),
) -> None:
    """View/update config (`fysearch.config.json`)."""
    cfg = load_config()
    changed = False

    if text_model:
        cfg.text_model = text_model
        changed = True
    if image_model:
        cfg.image_model = image_model
        changed = True
    if dataset_path:
        cfg.dataset_path = dataset_path
        changed = True
    if ocr_languages:
        cfg.ocr_languages = ocr_languages
        changed = True
    if embedding_dim:
        cfg.embedding_dim = embedding_dim
        changed = True
    if max_workers:
        cfg.max_workers = max_workers
        changed = True

    if changed:
        save_config(cfg)

    if show:
        table = Table(title="fysearch config")
        table.add_column("key")
        table.add_column("value")
        for k, v in config_to_table(cfg):
            table.add_row(str(k), str(v))
        console.print(table)


@app.command()
def ingest(path: Path = typer.Argument(..., exists=True)) -> None:
    """Ingest a file or directory into the local store."""
    conn = connect()
    init_db(conn)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Ingesting files...", total=None)
        results = ingest_path(conn, path)
        progress.update(task, completed=len(results), total=len(results))

    console.print(f"Ingested: {len(results)} files")
    conn.close()


@app.command()
def extract() -> None:
    """Extract text from all ingested documents (PDF text + optional OCR)."""
    paths = get_paths()
    conn = connect()
    init_db(conn)

    docs = list(list_documents(conn))
    conn.close()  # Close main connection — workers will open their own

    count = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Extracting text...", total=len(docs))

        # Prepare arguments for workers — include db_path as the 4th element
        work_items = [
            (row["doc_id"], str(row["stored_path"]), row["media_type"], str(paths.db_path))
            for row in docs
        ]

        # Use ThreadPoolExecutor for extraction - saturate all 8 logical processors
        # (OCR is I/O + CPU, threads avoid process overhead)
        cfg = load_config()
        max_w = cfg.effective_max_workers  # Will be 8 on your system

        with ThreadPoolExecutor(max_workers=max_w) as executor:
            futures = [executor.submit(_extract_worker, item) for item in work_items]

            for future in as_completed(futures):
                try:
                    if future.result():
                        count += 1
                except Exception as e:
                    # Log error but continue
                    console.print(f"[red]Error extracting doc:[/red] {e}")
                finally:
                    progress.advance(task)

    console.print(f"Extracted text for {count} documents")


def _get_index(dim: int, prefer_faiss: bool) -> object:
    if prefer_faiss:
        try:
            return FaissIndex(dim)
        except Exception:
            return BruteForceIndex(dim)
    return BruteForceIndex(dim)


@app.command(name="build-index")
def build_index(
    modality: str = typer.Option("text", help="Which index to build: text|image"),
    prefer_faiss: bool = typer.Option(True, help="Use FAISS if installed"),
) -> None:
    """Build an index and persist doc_ids+vectors as npz."""
    cfg = load_config()
    modality = modality.strip().lower()
    if modality not in {"text", "image"}:
        raise typer.BadParameter("modality must be: text|image")

    conn = connect()
    init_db(conn)

    if modality == "text":
        embedder = maybe_text_embedder(cfg.text_model)
        if embedder is None:
            raise typer.BadParameter("Config.text_model is empty")

        # Load extracted text
        rows = conn.execute(
            """
            SELECT d.doc_id, t.text
            FROM documents d
            JOIN extracted_text t ON t.doc_id = d.doc_id
            WHERE t.text IS NOT NULL
              AND length(trim(t.text)) > 0
              AND (
                -- Include per-page docs (they have #page= in original_path)
                d.original_path LIKE '%#page=%'
                -- Include non-PDF docs
                OR d.media_type NOT IN ('pdf', 'image')
                -- Include PDFs without per-page docs
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

        doc_ids: list[str] = []
        vectors: list[np.ndarray] = []

        # Filter rows with valid text
        valid_rows = []
        for r in rows:
            text = (r["text"] or "").strip()
            if text:
                valid_rows.append(r)

        # Batch processing — larger batches for better CPU utilization
        batch_size = 256

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Embedding text...", total=len(valid_rows))

            for i in range(0, len(valid_rows), batch_size):
                batch_rows = valid_rows[i : i + batch_size]
                batch_texts = [(r["text"] or "").strip() for r in batch_rows]

                # Embed batch
                batch_vecs = embedder.embed_batch(batch_texts, batch_size=len(batch_texts))

                for r, vec in zip(batch_rows, batch_vecs):
                    doc_ids.append(r["doc_id"])
                    vectors.append(vec)

                progress.advance(task, advance=len(batch_rows))

        if not vectors:
            console.print("No text to index yet. Run `fysearch extract` and ensure OCR/PDF text exists.")
            return

        mat = np.stack(vectors, axis=0).astype(np.float32)
        dim = mat.shape[1]

        idx = _get_index(dim=dim, prefer_faiss=prefer_faiss)
        idx.add(doc_ids, mat)

        paths = get_paths()
        out_path = paths.index_dir / "text_index.npz"
        paths.index_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, doc_ids=np.asarray(doc_ids), vectors=mat)
        console.print(f"Wrote index: {out_path}")
        return

    # modality == "image"
    embedder = maybe_image_embedder(cfg.image_model)
    if embedder is None:
        raise typer.BadParameter("Config.image_model is empty (set it via: fysearch config --image-model <model>")

    rows = conn.execute(
        """
        SELECT doc_id, stored_path
        FROM documents
        WHERE media_type = 'image'
        """
    ).fetchall()

    doc_ids: list[str] = []
    vectors: list[np.ndarray] = []

    rows = list(rows)
    batch_size = 128

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Embedding images...", total=len(rows))

        for i in range(0, len(rows), batch_size):
            batch_rows = rows[i : i + batch_size]
            batch_paths = [r["stored_path"] for r in batch_rows]

            try:
                batch_vecs = embedder.embed_batch(batch_paths, batch_size=len(batch_paths))

                for r, vec in zip(batch_rows, batch_vecs):
                    doc_ids.append(r["doc_id"])
                    vectors.append(vec)
            except Exception as e:
                console.print(f"[red]Error embedding batch {i}:[/red] {e}")

            progress.advance(task, advance=len(batch_rows))

    if not vectors:
        console.print("No images found to index.")
        return

    mat = np.stack(vectors, axis=0).astype(np.float32)
    dim = mat.shape[1]
    idx = _get_index(dim=dim, prefer_faiss=prefer_faiss)
    idx.add(doc_ids, mat)

    paths = get_paths()
    out_path = paths.index_dir / "image_index.npz"
    paths.index_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, doc_ids=np.asarray(doc_ids), vectors=mat)
    console.print(f"Wrote index: {out_path}")


@app.command(name="search-text")
def search_text(
    query: str,
    top_k: int = typer.Option(5),
    prefer_faiss: bool = typer.Option(True),
    modality: str = typer.Option("text", help="Search over: text|image (text->image when image index exists)"),
) -> None:
    """Semantic search over text, or text->image when modality=image."""
    cfg = load_config()
    embedder = maybe_text_embedder(cfg.text_model)
    if embedder is None:
        raise typer.BadParameter("Config.text_model is empty")

    paths = get_paths()
    modality = modality.strip().lower()
    if modality not in {"text", "image"}:
        raise typer.BadParameter("modality must be: text|image")

    npz = paths.index_dir / ("text_index.npz" if modality == "text" else "image_index.npz")
    if not npz.exists():
        raise typer.BadParameter("Index not found. Run: fysearch build-index")

    data = np.load(npz, allow_pickle=True)
    doc_ids = [str(x) for x in data["doc_ids"]]
    vectors = data["vectors"].astype(np.float32)

    dim = vectors.shape[1]
    idx = _get_index(dim=dim, prefer_faiss=prefer_faiss)
    idx.add(doc_ids, vectors)

    q = embedder.embed(query).astype(np.float32)
    hits = idx.search(q, top_k)

    conn = connect()
    init_db(conn)

    title = f"Results for: {query}" if modality == "text" else f"Text→Image results for: {query}"
    table = Table(title=title)
    table.add_column("score")
    table.add_column("doc_id")
    table.add_column("page")
    table.add_column("method")
    table.add_column("media")
    table.add_column("original_path")

    for hit in hits:
        row = conn.execute(
            """
            SELECT d.original_path, d.media_type, t.method
            FROM documents d
            LEFT JOIN extracted_text t ON t.doc_id = d.doc_id
            WHERE d.doc_id = ?
            """,
            (hit.doc_id,),
        ).fetchone()
        if not row:
            continue

        original_path = str(row["original_path"])
        page = ""
        marker = "#page="
        if marker in original_path:
            try:
                page = original_path.split(marker, 1)[1].strip()
            except Exception:
                page = ""
        method = str(row["method"] or "")
        media = str(row["media_type"] or "")
        table.add_row(f"{hit.score:.4f}", hit.doc_id[:12], page, method, media, original_path)

    console.print(table)
    conn.close()


@app.command(name="search-image")
def search_image(
    image: Path = typer.Argument(..., exists=True),
    top_k: int = typer.Option(5),
    prefer_faiss: bool = typer.Option(True),
) -> None:
    """Image→Image search using the image index."""
    cfg = load_config()
    embedder = maybe_image_embedder(cfg.image_model)
    if embedder is None:
        raise typer.BadParameter("Config.image_model is empty")

    paths = get_paths()
    npz = paths.index_dir / "image_index.npz"
    if not npz.exists():
        raise typer.BadParameter("Image index not found. Run: fysearch build-index --modality image")

    data = np.load(npz, allow_pickle=True)
    doc_ids = [str(x) for x in data["doc_ids"]]
    vectors = data["vectors"].astype(np.float32)

    dim = vectors.shape[1]
    idx = _get_index(dim=dim, prefer_faiss=prefer_faiss)
    idx.add(doc_ids, vectors)

    q = embedder.embed(str(image)).astype(np.float32)
    hits = idx.search(q, top_k)

    conn = connect()
    init_db(conn)

    table = Table(title=f"Image results for: {image}")
    table.add_column("score")
    table.add_column("doc_id")
    table.add_column("stored_path")

    for hit in hits:
        row = conn.execute("SELECT stored_path FROM documents WHERE doc_id = ?", (hit.doc_id,)).fetchone()
        stored = row["stored_path"] if row else "?"
        table.add_row(f"{hit.score:.4f}", hit.doc_id[:12], stored)

    console.print(table)
    conn.close()


@app.command()
def web(
    host: str = typer.Option("0.0.0.0", help="Bind host (0.0.0.0 for WSL access from Windows)"),
    port: int = typer.Option(5000, help="Bind port"),
    debug: bool = typer.Option(False, help="Enable Flask debug mode"),
) -> None:
    """Run a local Flask UI to view/search results."""
    try:
        from .webapp import create_app
    except Exception as e:
        raise typer.BadParameter(str(e))

    console.print(f"[bold green]Starting FYSearch web UI[/bold green]")
    console.print(f"  → http://{host}:{port}")
    if host == "0.0.0.0":
        console.print(f"  → http://127.0.0.1:{port}  (localhost)")
    app_ = create_app()
    app_.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    app()
