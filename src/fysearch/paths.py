from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


def is_wsl() -> bool:
    """Detect if running under Windows Subsystem for Linux."""
    if sys.platform != "linux":
        return False
    try:
        with open("/proc/version", "r") as f:
            return "microsoft" in f.read().lower()
    except (OSError, IOError):
        return False


def normalize_path(p: str) -> str:
    """Convert Windows paths to WSL-compatible paths when running under WSL.
    
    Handles:
      - C:\\Users\\... → /mnt/c/Users/...
      - C:/Users/...  → /mnt/c/Users/...
      - /mnt/c/...    → unchanged (already WSL format)
      - /home/...     → unchanged (native Linux path)
    """
    p = p.strip()
    if not p:
        return p

    if is_wsl() and len(p) >= 3 and p[1] == ':' and p[2] in ('\\', '/'):
        drive = p[0].lower()
        rest = p[3:].replace('\\', '/')
        return f"/mnt/{drive}/{rest}"

    return p


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def input_dir(self) -> Path:
        return self.data_dir / "input"

    @property
    def store_dir(self) -> Path:
        return self.data_dir / "store"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def db_dir(self) -> Path:
        return self.data_dir / "db"

    @property
    def db_path(self) -> Path:
        return self.db_dir / "fysearch.sqlite3"

    @property
    def config_path(self) -> Path:
        return self.root / "fysearch.config.json"


def get_project_root() -> Path:
    # Assumes CLI is run from repo root (or a subdir). Walk upward until we find pyproject.
    current = Path.cwd().resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return current


def get_paths() -> ProjectPaths:
    return ProjectPaths(root=get_project_root())
