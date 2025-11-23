"""Local LLM inference using Mistral 7B GGUF via llama-cpp-python."""

from llama_cpp import Llama
from typing import List, Dict, Optional
import logging
import os

logger = logging.getLogger(__name__)


class LocalLLM:
    """Wrapper for local Mistral 7B GGUF model."""
    
    def __init__(
        self,
        model_path: str = "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        n_ctx: int = 4096,
        n_threads: int = 4
    ):
        """
        Initialize local LLM.
        
        Args:
            model_path: Path to GGUF model file
            n_ctx: Context window size
            n_threads: Number of threads for inference
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                "Please download the model first. See models/README.md"
            )
        
        logger.info(f"Loading LLM from {model_path}")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            verbose=False
        )
        logger.info("LLM loaded successfully")
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: List[str] = None
    ) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            stop: Stop sequences
            
        Returns:
            Generated text
        """
        if stop is None:
            stop = ["</s>", "\n\n\n"]
        
        response = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            echo=False
        )
        
        return response['choices'][0]['text'].strip()
    
    def generate_with_citations(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, any]],
        max_tokens: int = 512
    ) -> Dict[str, any]:
        """
        Generate answer with citations from retrieved chunks.
        
        Args:
            query: User query
            retrieved_chunks: List of retrieved chunk dictionaries
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dictionary with 'answer' and 'citations'
        """
        # Build context from retrieved chunks
        context_parts = []
        for i, result in enumerate(retrieved_chunks, 1):
            chunk_text = result['chunk']['text']
            metadata = result.get('metadata', {})
            context_parts.append(f"[{i}] {chunk_text}")
        
        context = "\n\n".join(context_parts)
        
        # Build prompt
        prompt = f"""You are a financial market intelligence assistant. Answer the user's question based on the provided context. Include citations [1], [2], etc. when referencing specific information.

Context:
{context}

Question: {query}

Answer:"""
        
        # Generate answer
        answer = self.generate(prompt, max_tokens=max_tokens)
        
        # Extract citations from answer
        citations = []
        for i in range(1, len(retrieved_chunks) + 1):
            if f"[{i}]" in answer:
                citations.append({
                    "citation_id": i,
                    "chunk": retrieved_chunks[i-1]['chunk'],
                    "metadata": retrieved_chunks[i-1].get('metadata', {})
                })
        
        return {
            "answer": answer,
            "citations": citations,
            "retrieved_chunks": retrieved_chunks
        }

