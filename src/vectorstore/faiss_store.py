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
            query_embedding = np.expand_dims(query_embedding, axis=0)

        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype("float32")

        distances, indices = self.index.search(query_embedding, k)

        results = []
        for rank, idx in enumerate(indices[0]):
            if idx < len(self.chunks):
                results.append({
                    "rank": rank + 1,
                    "distance": float(distances[0][rank]),
                    "chunk": self.chunks[idx],
                    "metadata": self.chunks[idx].get("metadata", {})
                })

        return results

    def save(self, index_path: str, chunks_path: str):
        """
        Save FAISS index and chunks metadata.

        Args:
            index_path: path to write FAISS index file
            chunks_path: path to write chunk metadata
        """
        faiss.write_index(self.index, index_path)

        with open(chunks_path, "wb") as f:
            pickle.dump(self.chunks, f)

    @classmethod
    def load(cls, index_path: str, chunks_path: str):
        """
        Load FAISS index + chunk metadata.

        Returns:
            FAISSStore instance
        """
        index = faiss.read_index(index_path)

        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)

        store = cls(index.d)
        store.index = index
        store.chunks = chunks

        return store
