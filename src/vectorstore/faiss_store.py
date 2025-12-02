"""FAISS IndexFlatL2 for fast vector similarity search."""

import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class FAISSStore:
    """FAISS vector store using IndexFlatL2 for exact similarity search."""
    
    def __init__(self, dimension: int = 1024):
        """
        Initialize FAISS index.
        
        Args:
            dimension: Embedding dimension (BGE-Large-en: 1024)
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.chunks = []  # Store original chunk texts and metadata
        logger.info(f"Initialized FAISS index with dimension {dimension}")
    
    def add_chunks(self, embeddings: np.ndarray, chunks: List[Dict[str, any]]):
        """
        Add chunks and their embeddings to the index.
        
        Args:
            embeddings: Numpy array of embeddings (shape: [num_chunks, dimension])
            chunks: List of chunk dictionaries with text and metadata
        """
        if len(embeddings) != len(chunks):
            raise ValueError(f"Mismatch: {len(embeddings)} embeddings vs {len(chunks)} chunks")
        
        # Ensure embeddings are float32
        embeddings = embeddings.astype('float32')
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        # Store chunks
        self.chunks.extend(chunks)
        
        logger.info(f"Added {len(chunks)} chunks to FAISS index. Total: {self.index.ntotal}")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, any]]:
        """
        Search for top-k most similar chunks.
        
        Args:
            query_embedding: Query embedding vector (shape: [dimension])
            k: Number of results to return
            
        Returns:
            List of dictionaries with 'chunk', 'score', and 'metadata'
        """
        if self.index.ntotal == 0:
            logger.warning("Index is empty")
            return []
        
        # Ensure query is float32 and reshape to [1, dimension]
        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        
        # Search
        distances, indices = self.index.search(query_embedding, min(k, self.index.ntotal))
        
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.chunks):
                results.append({
                    "chunk": self.chunks[idx],
                    "score": float(distance),
                    "metadata": self.chunks[idx].get("metadata", {})
                })
        
        return results
    
    def save(self, index_path: str = "vectorstore/faiss.index", chunks_path: str = "vectorstore/chunks.pkl"):
        """Save FAISS index and chunks to disk."""
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        
        faiss.write_index(self.index, index_path)
        with open(chunks_path, 'wb') as f:
            pickle.dump(self.chunks, f)
        
        logger.info(f"Saved FAISS index to {index_path} and chunks to {chunks_path}")
    
    def load(self, index_path: str = "vectorstore/faiss.index", chunks_path: str = "vectorstore/chunks.pkl"):
        """Load FAISS index and chunks from disk."""
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found: {index_path}")
        if not os.path.exists(chunks_path):
            raise FileNotFoundError(f"Chunks file not found: {chunks_path}")
        
        self.index = faiss.read_index(index_path)
        with open(chunks_path, 'rb') as f:
            self.chunks = pickle.load(f)
        
        logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors from {index_path}")

