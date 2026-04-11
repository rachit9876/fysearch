from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from .paths import get_paths


@dataclass(frozen=True)
class Document:
    doc_id: str
    original_path: str
    stored_path: str
    media_type: str  # e.g. "pdf" | "image" | "unknown"
    created_at: str


def connect(db_path: Optional[Path] = None) -> sqlite3.Connection:
    paths = get_paths()
    db_path = db_path or paths.db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # Add timeout for concurrent access and enable WAL mode for better concurrency
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    # Enable WAL mode for better concurrent read/write performance
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            original_path TEXT NOT NULL,
            stored_path TEXT NOT NULL,
            media_type TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS extracted_text (
            doc_id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            method TEXT NOT NULL, -- "pdf_text" | "ocr"
            confidence REAL,
            language TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS embeddings (
            doc_id TEXT PRIMARY KEY,
            modality TEXT NOT NULL, -- "text" | "image"
            dim INTEGER NOT NULL,
            vector BLOB NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
        );
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS search_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            kind TEXT NOT NULL, -- "text" | "image"
            query_text TEXT,
            target_modality TEXT, -- "text" | "image" (when kind=text)
            top_k INTEGER,
            uploaded_path TEXT
        );
        """
    )
    conn.commit()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def add_search_history(
    conn: sqlite3.Connection,
    *,
    kind: str,
    query_text: Optional[str],
    target_modality: Optional[str],
    top_k: Optional[int],
    uploaded_path: Optional[str],
    created_at: Optional[str] = None,
) -> None:
    init_db(conn)
    conn.execute(
        """
        INSERT INTO search_history (created_at, kind, query_text, target_modality, top_k, uploaded_path)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            created_at or _now_iso(),
            kind,
            query_text,
            target_modality,
            top_k,
            uploaded_path,
        ),
    )
    conn.commit()


def list_search_history(conn: sqlite3.Connection, limit: int = 25) -> list[sqlite3.Row]:
    init_db(conn)
    return conn.execute(
        "SELECT * FROM search_history ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()


def clear_search_history(conn: sqlite3.Connection) -> list[str]:
    """Clear search history and return uploaded paths that were referenced."""
    init_db(conn)
    rows = conn.execute(
        "SELECT uploaded_path FROM search_history WHERE uploaded_path IS NOT NULL AND uploaded_path != ''"
    ).fetchall()
    uploaded_paths = [str(r["uploaded_path"]) for r in rows]
    conn.execute("DELETE FROM search_history")
    conn.commit()
    return uploaded_paths


def upsert_document(conn: sqlite3.Connection, doc: Document) -> None:
    conn.execute(
        """
        INSERT INTO documents (doc_id, original_path, stored_path, media_type, created_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(doc_id) DO UPDATE SET
            original_path=excluded.original_path,
            stored_path=excluded.stored_path,
            media_type=excluded.media_type;
        """,
        (doc.doc_id, doc.original_path, doc.stored_path, doc.media_type, doc.created_at),
    )
    conn.commit()


def list_documents(conn: sqlite3.Connection) -> Iterable[sqlite3.Row]:
    return conn.execute("SELECT * FROM documents ORDER BY created_at DESC")


def get_document(conn: sqlite3.Connection, doc_id: str) -> Optional[sqlite3.Row]:
    cur = conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,))
    return cur.fetchone()


def upsert_extracted_text(
    conn: sqlite3.Connection,
    doc_id: str,
    text: str,
    method: str,
    confidence: Optional[float],
    language: Optional[str],
    created_at: str,
) -> None:
    conn.execute(
        """
        INSERT INTO extracted_text (doc_id, text, method, confidence, language, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(doc_id) DO UPDATE SET
            text=excluded.text,
            method=excluded.method,
            confidence=excluded.confidence,
            language=excluded.language;
        """,
        (doc_id, text, method, confidence, language, created_at),
    )
    conn.commit()


def get_extracted_text(conn: sqlite3.Connection, doc_id: str) -> Optional[sqlite3.Row]:
    cur = conn.execute("SELECT * FROM extracted_text WHERE doc_id = ?", (doc_id,))
    return cur.fetchone()
