import faiss
import numpy as np
import pickle
from typing import List, Dict, Any


class FAISSStore:
    def __init__(self, dimension: int):
        """
        Initialize FAISS index.

        Args:
            dimension: Embedding dimension (e.g., 1024 for BGE-large)
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.chunks: List[Dict[str, Any]] = []

    def add_chunks(self, embeddings: np.ndarray, chunks: List[Dict[str, Any]]):
        """
        Add embeddings and corresponding chunks to index.

        Args:
            embeddings: numpy array of vectors
            chunks: list of chunk dictionaries
        """
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype("float32")

        self.index.add(embeddings)
        self.chunks = chunks

    def search(self, query_embedding: np.ndarray, k: int = 5):
        """
        Search FAISS index for nearest chunks.

        Args:
            query_embedding: vector from embedder.embed_query()
            k: top K results to return
        """
        if query_embedding.ndim == 1:
