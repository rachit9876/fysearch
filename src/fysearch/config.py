from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .paths import get_paths


@dataclass
class Config:
    # Model identifiers or local paths (for offline use).
    text_model: str = "sentence-transformers/clip-ViT-B-32"
    image_model: str = "sentence-transformers/clip-ViT-B-32"

    # Optional dataset folder path (e.g. where your test files live)
    dataset_path: str = ""

    # OCR languages for tesseract (e.g. "eng", "hin", "eng+hin").
    # Requires matching system language data packages.
    ocr_languages: str = "eng"

    # Embedding vector size (used for index creation; must match model output).
    # CLIP ViT-B/32 produces 512-dimensional vectors.
    embedding_dim: int = 512

    # Maximum number of parallel workers for CPU-intensive tasks.
    # 0 = auto-detect from os.cpu_count() (recommended).
    # Set higher for better CPU utilization (e.g., cpu_count * 2 for I/O-bound tasks)
    max_workers: int = 0

    @property
    def effective_max_workers(self) -> int:
        """Resolved worker count: if 0, auto-detect from CPU count."""
        if self.max_workers > 0:
            return self.max_workers
        cpu_count = os.cpu_count() or 4
        # Use all logical processors (hyperthreading) for maximum throughput
        # For 4 cores / 8 threads, this returns 8
        return cpu_count


def load_config() -> Config:
    paths = get_paths()
    if not paths.config_path.exists():
        return Config()
    data = json.loads(paths.config_path.read_text(encoding="utf-8"))
    # Filter out keys that don't exist in Config to prevent errors on old configs
    valid_keys = {f.name for f in Config.__dataclass_fields__.values()}
    filtered = {k: v for k, v in data.items() if k in valid_keys}
    return Config(**filtered)


def save_config(config: Config) -> None:
    paths = get_paths()
    paths.config_path.write_text(json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8")


def config_to_table(config: Config) -> list[tuple[str, Any]]:
    return [
        ("text_model", config.text_model),
        ("image_model", config.image_model),
        ("dataset_path", config.dataset_path),
        ("ocr_languages", config.ocr_languages),
        ("embedding_dim", config.embedding_dim),
        ("max_workers", f"{config.max_workers} (effective: {config.effective_max_workers})"),
    ]
