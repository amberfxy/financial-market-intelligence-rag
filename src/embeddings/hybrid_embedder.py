"""Hybrid embedding approach combining BGE and Word2Vec for different content types."""

import numpy as np
from typing import List, Union, Dict, Optional
import logging

from src.embeddings.embedder import BGEEmbedder
from src.embeddings.word2vec_embedder import Word2VecEmbedder

logger = logging.getLogger(__name__)


class HybridEmbedder:
    """
    Hybrid embedder that uses different strategies for different content types:
    - BGE for long-form content and general semantic understanding
    - Word2Vec for headlines and dense information
    """
    
    def __init__(
        self,
        bge_embedder: Optional[BGEEmbedder] = None,
        w2v_embedder: Optional[Word2VecEmbedder] = None,
        use_bge_for_long: bool = True,
        long_text_threshold: int = 100  # Characters
    ):
        """
        Initialize hybrid embedder.
        
        Args:
            bge_embedder: BGE embedder instance
            w2v_embedder: Word2Vec embedder instance
            use_bge_for_long: Use BGE for longer texts
            long_text_threshold: Character threshold for using BGE vs Word2Vec
        """
        self.bge_embedder = bge_embedder or BGEEmbedder()
        self.w2v_embedder = w2v_embedder
        self.use_bge_for_long = use_bge_for_long
        self.long_text_threshold = long_text_threshold
        
        logger.info("Hybrid embedder initialized")
    
    def embed_text(
        self,
        text: str,
        content_type: str = "auto",
        use_w2v: Optional[bool] = None
    ) -> Dict[str, np.ndarray]:
        """
        Embed text using appropriate strategy.
        
        Args:
            text: Input text
            content_type: Type of content ("headline", "article", "auto")
            use_w2v: Force use of Word2Vec (overrides auto-detection)
            
        Returns:
            Dictionary with embeddings: {"embedding": ..., "strategy": "bge"|"w2v"|"hybrid"}
        """
        # Determine strategy
        if use_w2v is True:
            strategy = "w2v"
        elif use_w2v is False:
            strategy = "bge"
        elif content_type == "headline":
            strategy = "w2v" if self.w2v_embedder else "bge"
        elif content_type == "article" or (self.use_bge_for_long and len(text) > self.long_text_threshold):
            strategy = "bge"
        elif self.w2v_embedder:
            # Auto: use Word2Vec for short dense texts, BGE for longer
            strategy = "w2v" if len(text) <= self.long_text_threshold else "bge"
        else:
            strategy = "bge"
        
        # Generate embedding
        if strategy == "w2v" and self.w2v_embedder:
            embedding = self.w2v_embedder.embed_text(text)
        elif strategy == "bge":
            embedding = self.bge_embedder.embed_texts([text])[0]
        else:
            # Fallback to BGE
            strategy = "bge"
            embedding = self.bge_embedder.embed_texts([text])[0]
        
        return {
            "embedding": embedding,
            "strategy": strategy,
            "dimension": len(embedding)
        }
    
    def embed_texts(
        self,
        texts: List[str],
        content_types: Optional[List[str]] = None,
        batch_size: int = 32
    ) -> Dict[str, any]:
        """
        Embed multiple texts using appropriate strategies.
        
        Args:
            texts: List of texts
            content_types: Optional list of content types per text
            batch_size: Batch size for BGE processing
            
        Returns:
            Dictionary with embeddings array, strategies, and dimensions
        """
        if content_types is None:
            content_types = ["auto"] * len(texts)
        
        embeddings = []
        strategies = []
        
        # Separate texts by strategy for batch processing
        bge_texts = []
        bge_indices = []
        w2v_texts = []
        w2v_indices = []
        
        for i, (text, content_type) in enumerate(zip(texts, content_types)):
            text_len = len(text)
            is_headline = content_type == "headline"
            
            if is_headline and self.w2v_embedder:
                w2v_texts.append(text)
                w2v_indices.append(i)
            elif self.use_bge_for_long and text_len > self.long_text_threshold:
                bge_texts.append(text)
                bge_indices.append(i)
            elif self.w2v_embedder and text_len <= self.long_text_threshold:
                w2v_texts.append(text)
                w2v_indices.append(i)
            else:
                bge_texts.append(text)
                bge_indices.append(i)
        
        # Process BGE texts in batches
        if bge_texts:
            logger.info(f"Processing {len(bge_texts)} texts with BGE")
            bge_embeddings = self.bge_embedder.embed_texts(bge_texts, batch_size=batch_size)
            for idx, emb in zip(bge_indices, bge_embeddings):
                embeddings.append(emb)
                strategies.append("bge")
        
        # Process Word2Vec texts
        if w2v_texts and self.w2v_embedder:
            logger.info(f"Processing {len(w2v_texts)} texts with Word2Vec")
            w2v_embeddings = self.w2v_embedder.embed_texts(w2v_texts)
            for idx, emb in zip(w2v_indices, w2v_embeddings):
                embeddings.append(emb)
                strategies.append("w2v")
        
        # Sort by original indices
        indexed = list(zip(range(len(texts)), embeddings, strategies))
        indexed.sort(key=lambda x: x[0])
        _, sorted_embeddings, sorted_strategies = zip(*indexed)
        
        return {
            "embeddings": np.array(sorted_embeddings),
            "strategies": sorted_strategies,
            "dimensions": [len(emb) for emb in sorted_embeddings]
        }
    
    def embed_query(self, query: str, use_w2v: bool = False) -> np.ndarray:
        """
        Embed a query. Defaults to BGE but can use Word2Vec.
        
        Args:
            query: Query text
            use_w2v: Use Word2Vec instead of BGE
            
        Returns:
            Query embedding
        """
        if use_w2v and self.w2v_embedder:
            return self.w2v_embedder.embed_text(query)
        else:
            return self.bge_embedder.embed_query(query)

