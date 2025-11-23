"""BGE-Large-en embedding model for semantic vector generation."""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union
import logging
import torch

logger = logging.getLogger(__name__)


class BGEEmbedder:
    """Wrapper for BGE-Large-en embedding model."""
    
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5", device: str = None):
        """
        Initialize the BGE embedding model.
        
        Args:
            model_name: HuggingFace model name
            device: Device to run on ('cuda', 'cpu', or None for auto)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Loading BGE model: {model_name} on {device}")
        self.model = SentenceTransformer(model_name, device=device)
        self.device = device
        logger.info("BGE model loaded successfully")
    
    def embed_texts(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Single text string or list of texts
            batch_size: Batch size for processing
            
        Returns:
            Numpy array of embeddings (shape: [num_texts, embedding_dim])
        """
        if isinstance(texts, str):
            texts = [texts]
        
        logger.info(f"Generating embeddings for {len(texts)} texts")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalize for cosine similarity
        )
        
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a query with instruction prefix for better retrieval.
        
        Args:
            query: Query text
            
        Returns:
            Query embedding vector
        """
        # BGE models use instruction prefix for queries
        instruction = "Represent this sentence for searching relevant passages: "
        query_with_instruction = instruction + query
        
        embedding = self.model.encode(
            query_with_instruction,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        return embedding

