"""Factory helpers for selecting PyTorch or ONNX BGE embedders."""

from __future__ import annotations

import logging
from typing import Any, Optional, Union

from src.embeddings.onnx_embedder import (
    DEFAULT_ONNX_DIR,
    ONNXBGEEmbedder,
    onnx_model_available,
)

logger = logging.getLogger(__name__)

# Runtime type includes both backends; annotated loosely to avoid importing torch
# until the PyTorch fallback path is actually needed.
BGEEmbedderLike = Any


def create_bge_embedder(
    prefer_onnx: bool = True,
    onnx_model_dir: str = DEFAULT_ONNX_DIR,
    model_name: str = "BAAI/bge-large-en-v1.5",
    device: Optional[str] = None,
) -> Union[ONNXBGEEmbedder, Any]:
    """
    Create a BGE embedder, preferring ONNX Runtime when an export is available.

    Falls back to the PyTorch sentence-transformers path if ONNX is unavailable.
    PyTorch / sentence-transformers are imported only on the fallback path.
    """
    if prefer_onnx and onnx_model_available(onnx_model_dir):
        try:
            embedder = ONNXBGEEmbedder(model_dir=onnx_model_dir)
            logger.info("Using ONNX Runtime BGE embedder")
            return embedder
        except Exception as exc:
            logger.warning(
                "Failed to load ONNX BGE embedder (%s). Falling back to PyTorch.",
                exc,
            )

    if prefer_onnx:
        logger.info(
            "ONNX BGE model not found at %s. Using PyTorch BGEEmbedder. "
            "Export with: python scripts/export_bge_onnx.py",
            onnx_model_dir,
        )

    from src.embeddings.embedder import BGEEmbedder

    return BGEEmbedder(model_name=model_name, device=device)
