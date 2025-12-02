"""Main RAG pipeline orchestrating retrieval and generation."""

import time
from typing import Dict, List, Optional, Union
import logging

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.embeddings.embedder import BGEEmbedder
from src.embeddings.hybrid_embedder import HybridEmbedder
from src.vectorstore.faiss_store import FAISSStore
from src.vectorstore.multi_source_store import MultiSourceFAISSStore
from src.rag.llm import LocalLLM

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Complete RAG pipeline for financial market intelligence."""
    
    def __init__(
        self,
        embedder: Union[BGEEmbedder, HybridEmbedder],
        vectorstore: Union[FAISSStore, MultiSourceFAISSStore],
        llm: LocalLLM,
        top_k: int = 5,
        balance_sources: bool = True,
        use_hybrid_embeddings: bool = False
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            embedder: BGE or Hybrid embedding model
            vectorstore: FAISS or MultiSourceFAISS vector store
            llm: Local LLM for generation
            top_k: Number of chunks to retrieve
            balance_sources: Balance citations across different data sources
            use_hybrid_embeddings: Use hybrid embedding strategy
        """
        self.embedder = embedder
        self.vectorstore = vectorstore
        self.llm = llm
        self.top_k = top_k
        self.balance_sources = balance_sources
        self.use_hybrid_embeddings = use_hybrid_embeddings
        
        logger.info(
            f"RAG pipeline initialized (balance_sources={balance_sources}, "
            f"use_hybrid={use_hybrid_embeddings})"
        )
    
    def query(self, query_text: str, top_k: Optional[int] = None) -> Dict[str, any]:
        """
        Process a query through the RAG pipeline.
        
        Args:
            query_text: User query
            top_k: Number of chunks to retrieve (overrides default)
            
        Returns:
            Dictionary with answer, citations, and metadata
        """
        start_time = time.time()
        
        if top_k is None:
            top_k = self.top_k
        
        # Step 1: Embed query
        if isinstance(self.embedder, HybridEmbedder):
            # Hybrid embedder can choose strategy based on query
            query_embedding_result = self.embedder.embed_text(
                query_text,
                content_type="auto",
                use_w2v=False  # Default to BGE for queries
            )
            query_embedding = query_embedding_result["embedding"]
            embedding_strategy = query_embedding_result.get("strategy", "bge")
        else:
            query_embedding = self.embedder.embed_query(query_text)
            embedding_strategy = "bge"
        
        embedding_time = time.time() - start_time
        
        # Step 2: Retrieve relevant chunks
        if isinstance(self.vectorstore, MultiSourceFAISSStore):
            # Use balanced retrieval for multi-source store
            retrieved_chunks = self.vectorstore.search(
                query_embedding,
                k=top_k,
                balance_sources=self.balance_sources
            )
        else:
            # Standard retrieval
            retrieved_chunks = self.vectorstore.search(query_embedding, k=top_k)
        
        retrieval_time = time.time() - start_time - embedding_time
        
        if not retrieved_chunks:
            return {
                "answer": "No relevant information found in the database.",
                "citations": [],
                "retrieved_chunks": [],
                "latency": {
                    "total": time.time() - start_time,
                    "embedding": embedding_time,
                    "retrieval": retrieval_time,
                    "generation": 0
                }
            }
        
        # Step 3: Generate answer with citations
        generation_start = time.time()
        result = self.llm.generate_with_citations(query_text, retrieved_chunks)
        generation_time = time.time() - generation_start
        
        # Add latency information
        result["latency"] = {
            "total": time.time() - start_time,
            "embedding": embedding_time,
            "retrieval": retrieval_time,
            "generation": generation_time
        }
        
        logger.info(
            f"Query processed in {result['latency']['total']:.2f}s "
            f"(embed: {embedding_time:.2f}s, retrieve: {retrieval_time:.2f}s, "
            f"generate: {generation_time:.2f}s)"
        )
        
        return result

