from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional, List

import numpy as np


def _configure_cpu_parallelism() -> None:
    """Configure ALL CPU parallelism knobs for maximum throughput on 4C/8T.

    Sets env vars for OpenMP, MKL, OpenBLAS, and HuggingFace tokenizers
    BEFORE any library reads them. Also configures PyTorch thread counts.
    """
    num_cores = os.cpu_count() or 8

    # These env vars must be set before numpy/torch/ONNX import in workers,
    # but since this module is imported early, we set them here.
    os.environ.setdefault("OMP_NUM_THREADS", str(num_cores))
    os.environ.setdefault("MKL_NUM_THREADS", str(num_cores))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(num_cores))
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(num_cores))
    os.environ.setdefault("NUMEXPR_MAX_THREADS", str(num_cores))
    # Avoid deadlocks in forked workers when using HF tokenizers
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

    try:
        import torch
        torch.set_num_threads(num_cores)
        try:
            torch.set_num_interop_threads(num_cores)
        except RuntimeError:
            pass  # Can only be set once; ignore if already set
    except ImportError:
        pass


# Configure on module load — must happen before any model inference.
_configure_cpu_parallelism()


def l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12
    return v / norm


@dataclass
class TextEmbedder:
    model_name_or_path: str
    _model: Any = field(default=None, init=False, repr=False)

    def _get_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("sentence-transformers not installed; install extras: pip install -e .[embeddings]") from e
        self._model = SentenceTransformer(self.model_name_or_path)
        return self._model

    def embed(self, text: str) -> np.ndarray:
        model = self._get_model()
        try:
            import torch
            with torch.inference_mode():
                vec = np.asarray(model.encode([text], normalize_embeddings=True)[0], dtype=np.float32)
        except ImportError:
            vec = np.asarray(model.encode([text], normalize_embeddings=True)[0], dtype=np.float32)
        return vec

    def embed_batch(self, texts: List[str], batch_size: int = 256) -> np.ndarray:
        """Embed multiple texts efficiently in batches. Returns array of shape (len(texts), dim)."""
        model = self._get_model()
        try:
            import torch
            with torch.inference_mode():
                vectors = model.encode(
                    texts,
                    normalize_embeddings=True,
                    batch_size=batch_size,
                    show_progress_bar=False,
                )
        except ImportError:
            vectors = model.encode(
                texts,
                normalize_embeddings=True,
                batch_size=batch_size,
                show_progress_bar=False,
            )
        return np.asarray(vectors, dtype=np.float32)


@dataclass
class ImageEmbedder:
    model_name_or_path: str
    _model: Any = field(default=None, init=False, repr=False)

    def _get_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("sentence-transformers not installed; install extras: pip install -e .[embeddings]") from e
        self._model = SentenceTransformer(self.model_name_or_path)
        return self._model

    def embed(self, image_path: str) -> np.ndarray:
        from PIL import Image

        model = self._get_model()
        img = Image.open(image_path).convert("RGB")
        try:
            import torch
            with torch.inference_mode():
                vec = np.asarray(model.encode([img], normalize_embeddings=True)[0], dtype=np.float32)
        except ImportError:
            vec = np.asarray(model.encode([img], normalize_embeddings=True)[0], dtype=np.float32)
        return vec

    def embed_batch(self, image_paths: List[str], batch_size: int = 128) -> np.ndarray:
        """Embed multiple images efficiently in batches. Returns array of shape (len(images), dim)."""
        from PIL import Image
        from concurrent.futures import ThreadPoolExecutor

        model = self._get_model()
        
        # Parallel image loading - use ALL logical processors for I/O
        num_loaders = min(os.cpu_count() or 8, len(image_paths))
        def load_image(path: str) -> Image.Image:
            return Image.open(path).convert("RGB")
        
        with ThreadPoolExecutor(max_workers=num_loaders) as executor:
            images = list(executor.map(load_image, image_paths))
        try:
            import torch
            with torch.inference_mode():
                vectors = model.encode(
                    images,
                    normalize_embeddings=True,
                    batch_size=batch_size,
                    show_progress_bar=False,
                )
        except ImportError:
            vectors = model.encode(
                images,
                normalize_embeddings=True,
                batch_size=batch_size,
                show_progress_bar=False,
            )
        return np.asarray(vectors, dtype=np.float32)


def maybe_text_embedder(model_name_or_path: str) -> Optional[TextEmbedder]:
    if not model_name_or_path:
        return None
    return TextEmbedder(model_name_or_path=model_name_or_path)


def maybe_image_embedder(model_name_or_path: str) -> Optional[ImageEmbedder]:
    if not model_name_or_path:
        return None
    return ImageEmbedder(model_name_or_path=model_name_or_path)
