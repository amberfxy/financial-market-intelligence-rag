"""Multi-source vector store with balanced retrieval from different data sources."""

import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Tuple, Optional
import logging
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


class MultiSourceFAISSStore:
    """
    FAISS vector store that supports multiple data sources and balanced retrieval.
    Prevents one data source from dominating citations.
    """
    
    def __init__(self, dimension: int = 1024):
        """
        Initialize multi-source FAISS index.
        
        Args:
            dimension: Embedding dimension
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.chunks = []  # Store original chunk texts and metadata
        self.source_indices = defaultdict(list)  # Map source -> list of chunk indices
        
        logger.info(f"Initialized multi-source FAISS index with dimension {dimension}")
    
    def add_chunks(self, embeddings: np.ndarray, chunks: List[Dict[str, any]]):
        """
        Add chunks and their embeddings to the index.
        Tracks source information for balanced retrieval.
        
        Args:
            embeddings: Numpy array of embeddings (shape: [num_chunks, dimension])
            chunks: List of chunk dictionaries with text and metadata
        """
        if len(embeddings) != len(chunks):
            raise ValueError(f"Mismatch: {len(embeddings)} embeddings vs {len(chunks)} chunks")
        
        # Ensure embeddings are float32
        embeddings = embeddings.astype('float32')
        
        # Get starting index for new chunks
        start_idx = self.index.ntotal
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        # Store chunks and track sources
        for i, chunk in enumerate(chunks):
            chunk_idx = start_idx + i
            self.chunks.append(chunk)
            
            # Extract source identifier from metadata
            metadata = chunk.get("metadata", {})
            source = self._get_source_identifier(metadata)
            
            # Track source indices
            self.source_indices[source].append(chunk_idx)
        
        logger.info(
            f"Added {len(chunks)} chunks to FAISS index. "
            f"Total: {self.index.ntotal}. "
            f"Sources: {list(self.source_indices.keys())}"
        )
    
    def _get_source_identifier(self, metadata: Dict) -> str:
        """
        Extract source identifier from metadata.
        
        Args:
            metadata: Chunk metadata
            
        Returns:
            Source identifier string
        """
        # Try different metadata fields for source identification
        if "source" in metadata:
            return str(metadata["source"])
        elif "source_type" in metadata:
            return str(metadata["source_type"])
        elif "data_source" in metadata:
            return str(metadata["data_source"])
        elif "headline" in metadata:
            # Use headline as source identifier (for headlines dataset)
            return "headlines"
        else:
            # Default: use first part of content type or "unknown"
            return metadata.get("content_type", "unknown")
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        balance_sources: bool = True,
        max_per_source: Optional[int] = None
    ) -> List[Dict[str, any]]:
        """
        Search for top-k chunks with optional source balancing.
        
        Args:
            query_embedding: Query embedding vector (shape: [dimension])
            k: Number of results to return
            balance_sources: If True, balance results across sources
            max_per_source: Maximum results per source (auto-calculated if None)
            
        Returns:
            List of dictionaries with 'chunk', 'score', and 'metadata'
        """
        if self.index.ntotal == 0:
            logger.warning("Index is empty")
            return []
        
        # Ensure query is float32 and reshape
        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        
        if not balance_sources:
            # Simple retrieval without source balancing
            return self._simple_search(query_embedding, k)
        
        # Balanced retrieval across sources
        if max_per_source is None:
            num_sources = len(self.source_indices)
            if num_sources > 0:
                max_per_source = max(1, k // num_sources)
            else:
                max_per_source = k
        
        return self._balanced_search(query_embedding, k, max_per_source)
    
    def _simple_search(
        self,
        query_embedding: np.ndarray,
        k: int
    ) -> List[Dict[str, any]]:
        """Simple search without source balancing."""
        distances, indices = self.index.search(query_embedding, min(k, self.index.ntotal))
        
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx < len(self.chunks):
                results.append({
                    "chunk": self.chunks[idx],
                    "score": float(distance),
                    "metadata": self.chunks[idx].get("metadata", {})
                })
        
        return results
    
    def _balanced_search(
        self,
        query_embedding: np.ndarray,
        k: int,
        max_per_source: int
    ) -> List[Dict[str, any]]:
        """
        Balanced search that retrieves from multiple sources.
        
        Strategy:
        1. Search top-k*2 to get candidate pool
        2. Group candidates by source
        3. Select up to max_per_source from each source
        4. Fill remaining slots from best candidates across all sources
        """
        # Get larger candidate pool
        candidate_k = min(k * 3, self.index.ntotal)
        distances, indices = self.index.search(query_embedding, candidate_k)
        
        # Build candidate list with source information
        candidates = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx >= len(self.chunks):
                continue
            
            chunk = self.chunks[idx]
            metadata = chunk.get("metadata", {})
            source = self._get_source_identifier(metadata)
            
            candidates.append({
                "chunk": chunk,
                "score": float(distance),
                "metadata": metadata,
                "source": source,
                "index": idx
            })
        
        if not candidates:
            return []
        
        # Group candidates by source
        source_groups = defaultdict(list)
        for candidate in candidates:
            source_groups[candidate["source"]].append(candidate)
        
        # Sort candidates within each source by score (lower is better for L2)
        for source in source_groups:
            source_groups[source].sort(key=lambda x: x["score"])
        
        # Select balanced results
        selected = []
        source_counts = Counter()
        
        # First pass: select up to max_per_source from each source
        for source, source_candidates in source_groups.items():
            selected_from_source = source_candidates[:max_per_source]
            selected.extend(selected_from_source)
            source_counts[source] = len(selected_from_source)
        
        # If we have enough results, sort and return top k
        if len(selected) >= k:
            selected.sort(key=lambda x: x["score"])
            selected = selected[:k]
        else:
            # Second pass: fill remaining slots from best candidates
            # Sort all candidates by score
            all_candidates_sorted = sorted(candidates, key=lambda x: x["score"])
            
            # Add candidates that aren't already selected
            for candidate in all_candidates_sorted:
                if len(selected) >= k:
                    break
                
                # Check if already selected
                already_selected = any(
                    s["index"] == candidate["index"] for s in selected
                )
                
                if not already_selected:
                    selected.append(candidate)
                    source_counts[candidate["source"]] += 1
        
        # Sort final results by score
        selected.sort(key=lambda x: x["score"])
        
        # Format results (remove internal fields)
        results = []
        for item in selected[:k]:
            results.append({
                "chunk": item["chunk"],
                "score": item["score"],
                "metadata": item["metadata"]
            })
        
        logger.debug(
            f"Balanced retrieval: {len(results)} results from {len(source_counts)} sources. "
            f"Source distribution: {dict(source_counts)}"
        )
        
        return results
    
    def get_source_statistics(self) -> Dict[str, int]:
        """Get statistics about sources in the index."""
        return {
            source: len(indices)
            for source, indices in self.source_indices.items()
        }
    
    def save(
        self,
        index_path: str = "vectorstore/faiss.index",
        chunks_path: str = "vectorstore/chunks.pkl",
        sources_path: str = "vectorstore/sources.pkl"
    ):
        """Save FAISS index, chunks, and source mappings to disk."""
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        
        faiss.write_index(self.index, index_path)
        
        with open(chunks_path, 'wb') as f:
            pickle.dump(self.chunks, f)
        
        with open(sources_path, 'wb') as f:
            pickle.dump(dict(self.source_indices), f)
        
        logger.info(
            f"Saved multi-source FAISS index to {index_path}, "
            f"chunks to {chunks_path}, and sources to {sources_path}"
        )
    
    def load(
        self,
        index_path: str = "vectorstore/faiss.index",
        chunks_path: str = "vectorstore/chunks.pkl",
        sources_path: Optional[str] = "vectorstore/sources.pkl"
    ):
        """Load FAISS index, chunks, and source mappings from disk."""
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found: {index_path}")
        if not os.path.exists(chunks_path):
            raise FileNotFoundError(f"Chunks file not found: {chunks_path}")
        
        self.index = faiss.read_index(index_path)
        
        with open(chunks_path, 'rb') as f:
            self.chunks = pickle.load(f)
        
        # Load source mappings if available
        if sources_path and os.path.exists(sources_path):
            with open(sources_path, 'rb') as f:
                self.source_indices = defaultdict(list, pickle.load(f))
        else:
            # Reconstruct source indices from chunks
            logger.info("Reconstructing source indices from chunks...")
            self.source_indices = defaultdict(list)
            for i, chunk in enumerate(self.chunks):
                metadata = chunk.get("metadata", {})
                source = self._get_source_identifier(metadata)
                self.source_indices[source].append(i)
        
        logger.info(
            f"Loaded multi-source FAISS index with {self.index.ntotal} vectors from {index_path}. "
            f"Sources: {list(self.source_indices.keys())}"
        )

