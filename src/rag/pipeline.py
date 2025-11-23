"""Main RAG pipeline orchestrating retrieval and generation."""

import time
from typing import Dict, List, Optional
import logging

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.embeddings.embedder import BGEEmbedder
from src.vectorstore.faiss_store import FAISSStore
from src.rag.llm import LocalLLM

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Complete RAG pipeline for financial market intelligence."""
    
    def __init__(
        self,
        embedder: BGEEmbedder,
        vectorstore: FAISSStore,
        llm: LocalLLM,
        top_k: int = 5
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            embedder: BGE embedding model
            vectorstore: FAISS vector store
            llm: Local LLM for generation
            top_k: Number of chunks to retrieve
        """
        self.embedder = embedder
        self.vectorstore = vectorstore
        self.llm = llm
        self.top_k = top_k
        logger.info("RAG pipeline initialized")
    
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
        query_embedding = self.embedder.embed_query(query_text)
        embedding_time = time.time() - start_time
        
        # Step 2: Retrieve relevant chunks
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

