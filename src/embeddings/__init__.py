"""Embedding generation using BGE-Large-en model."""

from src.embeddings.factory import create_bge_embedder
from src.embeddings.onnx_embedder import ONNXBGEEmbedder, onnx_model_available

__all__ = [
    "BGEEmbedder",
    "ONNXBGEEmbedder",
    "create_bge_embedder",
    "onnx_model_available",
]


def __getattr__(name: str):
    # Lazy-load PyTorch embedder so ONNX-only paths stay lightweight.
    if name == "BGEEmbedder":
        from src.embeddings.embedder import BGEEmbedder

        return BGEEmbedder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
