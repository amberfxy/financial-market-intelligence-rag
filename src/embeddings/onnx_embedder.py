"""BGE-Large-en embedding via ONNX Runtime for faster CPU inference."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

DEFAULT_ONNX_DIR = "models/bge-large-en-v1.5-onnx"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def onnx_model_available(model_dir: Union[str, Path] = DEFAULT_ONNX_DIR) -> bool:
    """Return True if an exported ONNX BGE model directory is ready to load."""
    model_path = Path(model_dir)
    return (model_path / "model.onnx").is_file() and (
        (model_path / "tokenizer.json").is_file()
        or (model_path / "tokenizer_config.json").is_file()
    )


class ONNXBGEEmbedder:
    """
    ONNX Runtime wrapper for BGE-Large-en-v1.5.

    Matches BGEEmbedder's public API (embed_texts / embed_query) so it can
    drop into the RAG pipeline and HybridEmbedder without changes.
    Uses CLS pooling + L2 normalization, same as sentence-transformers BGE.
    """

    def __init__(
        self,
        model_dir: str = DEFAULT_ONNX_DIR,
        max_length: int = 512,
        providers: Optional[List[str]] = None,
    ):
        """
        Initialize ONNX BGE embedder.

        Args:
            model_dir: Directory containing model.onnx and tokenizer files
            max_length: Max sequence length for tokenization
            providers: ONNX Runtime execution providers (auto-detected if None)
        """
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required for ONNXBGEEmbedder. "
                "Install with: pip install onnxruntime"
            ) from exc

        model_path = Path(model_dir)
        onnx_file = model_path / "model.onnx"
        if not onnx_file.is_file():
            raise FileNotFoundError(
                f"ONNX model not found: {onnx_file}\n"
                "Export it first with: python scripts/export_bge_onnx.py"
            )

        if providers is None:
            available = ort.get_available_providers()
            preferred = [
                "CUDAExecutionProvider",
                "CoreMLExecutionProvider",
                "CPUExecutionProvider",
            ]
            providers = [p for p in preferred if p in available]
            if not providers:
                providers = ["CPUExecutionProvider"]

        logger.info(f"Loading ONNX BGE model from {model_dir} (providers={providers})")
        self.session = ort.InferenceSession(str(onnx_file), providers=providers)
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self.max_length = max_length
        self.model_dir = str(model_path)
        self.input_names = {inp.name for inp in self.session.get_inputs()}
        active = self.session.get_providers()
        if "CUDAExecutionProvider" in active:
            self.device = "cuda"
        elif "CoreMLExecutionProvider" in active:
            self.device = "coreml"
        else:
            self.device = "cpu"
        logger.info(
            f"ONNX BGE model loaded successfully "
            f"(active providers={active})"
        )
    def _pool_and_normalize(self, last_hidden_state: np.ndarray) -> np.ndarray:
        """CLS token pooling followed by L2 normalization."""
        cls = last_hidden_state[:, 0, :].astype(np.float32)
        norms = np.linalg.norm(cls, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-12, a_max=None)
        return cls / norms

    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        """Tokenize and run a single ONNX batch."""
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )

        ort_inputs = {}
        if "input_ids" in self.input_names:
            ort_inputs["input_ids"] = encoded["input_ids"].astype(np.int64)
        if "attention_mask" in self.input_names:
            ort_inputs["attention_mask"] = encoded["attention_mask"].astype(np.int64)
        if "token_type_ids" in self.input_names:
            if "token_type_ids" in encoded:
                ort_inputs["token_type_ids"] = encoded["token_type_ids"].astype(np.int64)
            else:
                ort_inputs["token_type_ids"] = np.zeros_like(
                    encoded["input_ids"], dtype=np.int64
                )

        outputs = self.session.run(None, ort_inputs)
        last_hidden_state = outputs[0]
        return self._pool_and_normalize(last_hidden_state)

    def embed_texts(
        self, texts: Union[str, List[str]], batch_size: int = 32
    ) -> np.ndarray:
        """
        Generate embeddings for text(s).

        Args:
            texts: Single text string or list of texts
            batch_size: Batch size for ONNX inference

        Returns:
            Numpy array of embeddings (shape: [num_texts, embedding_dim])
        """
        if isinstance(texts, str):
            texts = [texts]

        logger.info(f"Generating ONNX embeddings for {len(texts)} texts")
        all_embeddings = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            all_embeddings.append(self._encode_batch(batch))

        return np.vstack(all_embeddings)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a query with BGE instruction prefix for retrieval.

        Args:
            query: Query text

        Returns:
            Query embedding vector (1D)
        """
        query_with_instruction = QUERY_INSTRUCTION + query
        return self._encode_batch([query_with_instruction])[0]
