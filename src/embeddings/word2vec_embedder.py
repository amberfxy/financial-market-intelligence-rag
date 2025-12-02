"""Word2Vec embedding model for dense information like headlines."""

import numpy as np
from typing import List, Union
import logging
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class Word2VecEmbedder:
    """Word2Vec-based embedder optimized for headlines and dense information."""
    
    def __init__(self, model=None, vector_size: int = 300, window: int = 5, 
                 min_count: int = 2, workers: int = 4, sg: int = 1):
        """
        Initialize Word2Vec embedder.
        
        Args:
            model: Pre-trained Word2Vec model (optional)
            vector_size: Dimensionality of word vectors
            window: Maximum distance between current and predicted word
            min_count: Minimum frequency for words to be included
            workers: Number of threads
            sg: Training algorithm (0=CBOW, 1=skip-gram)
        """
        self.model = model
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.sg = sg
        
        if model is not None:
            logger.info("Using pre-trained Word2Vec model")
            self.vector_size = model.vector_size
        else:
            logger.info(f"Word2Vec embedder initialized (vector_size={vector_size})")
    
    def train(self, texts: List[str], epochs: int = 10):
        """
        Train Word2Vec model on a corpus.
        
        Args:
            texts: List of text strings
            epochs: Number of training epochs
        """
        logger.info(f"Training Word2Vec model on {len(texts)} texts...")
        
        # Preprocess texts
        sentences = [simple_preprocess(text, min_len=2, max_len=100) for text in texts]
        
        # Filter empty sentences
        sentences = [s for s in sentences if len(s) > 0]
        
        if not sentences:
            raise ValueError("No valid sentences found for training")
        
        # Train model
        self.model = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            sg=self.sg,
            epochs=epochs
        )
        
        logger.info(f"Word2Vec model trained. Vocabulary size: {len(self.model.wv)}")
    
    def _text_to_words(self, text: str) -> List[str]:
        """Convert text to list of words."""
        return simple_preprocess(text, min_len=2, max_len=100)
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single text by averaging word vectors.
        
        Args:
            text: Input text
            
        Returns:
            Average word vector embedding
        """
        if self.model is None:
            raise ValueError("Model not initialized. Train or load a model first.")
        
        words = self._text_to_words(text)
        
        if not words:
            # Return zero vector if no valid words
            return np.zeros(self.vector_size, dtype=np.float32)
        
        # Get word vectors
        word_vectors = []
        for word in words:
            if word in self.model.wv:
                word_vectors.append(self.model.wv[word])
        
        if not word_vectors:
            # Return zero vector if no words in vocabulary
            return np.zeros(self.vector_size, dtype=np.float32)
        
        # Average word vectors
        embedding = np.mean(word_vectors, axis=0).astype(np.float32)
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def embed_texts(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: Single text string or list of texts
            
        Returns:
            Numpy array of embeddings (shape: [num_texts, vector_size])
        """
        if isinstance(texts, str):
            texts = [texts]
        
        logger.info(f"Generating Word2Vec embeddings for {len(texts)} texts")
        
        embeddings = []
        for text in texts:
            embedding = self.embed_text(text)
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def save(self, filepath: str):
        """Save the Word2Vec model."""
        if self.model is None:
            raise ValueError("No model to save")
        self.model.save(filepath)
        logger.info(f"Word2Vec model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load a pre-trained Word2Vec model."""
        from gensim.models import Word2Vec
        self.model = Word2Vec.load(filepath)
        self.vector_size = self.model.vector_size
        logger.info(f"Word2Vec model loaded from {filepath}")

