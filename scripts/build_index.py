"""Script to build FAISS index from processed data.
Supports both basic and enhanced features (Word2Vec, hybrid embeddings, multi-source).
"""

import sys
import os
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import logging
import argparse

from src.data.loader import load_kaggle_dataset, preprocess_data
from src.chunking.chunker import chunk_dataframe
from src.embeddings.embedder import BGEEmbedder
from src.vectorstore.faiss_store import FAISSStore

# Optional imports for enhanced features
try:
    from src.embeddings.word2vec_embedder import Word2VecEmbedder
    from src.embeddings.hybrid_embedder import HybridEmbedder
    from src.vectorstore.multi_source_store import MultiSourceFAISSStore
    ENHANCED_FEATURES_AVAILABLE = True
except ImportError:
    ENHANCED_FEATURES_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_index(
    data_path: str = "data/raw",
    output_index: str = "vectorstore/faiss.index",
    output_chunks: str = "vectorstore/chunks.pkl",
    output_sources: Optional[str] = None,
    use_word2vec: bool = False,
    use_hybrid: bool = False,
    use_multi_source: bool = False,
    w2v_model_path: Optional[str] = None,
    train_w2v: bool = False,
    w2v_epochs: int = 10
):
    """
    Build FAISS index from dataset.
    
    Args:
        data_path: Path to raw data directory
        output_index: Output index file path
        output_chunks: Output chunks file path
        output_sources: Output sources file path (for multi-source, optional)
        use_word2vec: Train/use Word2Vec embeddings (requires enhanced features)
        use_hybrid: Use hybrid embedding approach (requires enhanced features)
        use_multi_source: Use multi-source aware storage (requires enhanced features)
        w2v_model_path: Path to pre-trained Word2Vec model (optional)
        train_w2v: Train Word2Vec model on dataset
        w2v_epochs: Number of epochs for Word2Vec training
    """
    # Check if enhanced features are requested but not available
    if (use_word2vec or use_hybrid or use_multi_source) and not ENHANCED_FEATURES_AVAILABLE:
        logger.warning("Enhanced features requested but not available. Using basic mode.")
        use_word2vec = False
        use_hybrid = False
        use_multi_source = False
    
    logger.info("Starting index building process...")
    
    # Step 1: Load and preprocess data
    logger.info("Loading dataset...")
    df = load_kaggle_dataset(data_path)
    df = preprocess_data(df)
    
    # Add source metadata if using multi-source
    if use_multi_source:
        if "source" not in df.columns and "source_type" not in df.columns:
            df["source_type"] = "headlines"
    
    # Step 2: Chunk documents
    logger.info("Chunking documents...")
    chunks = chunk_dataframe(df, text_column="News Headline", max_tokens=250)
    
    # Add source metadata to chunks if using multi-source
    if use_multi_source:
        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            source_idx = metadata.get("source_index", 0)
            if source_idx < len(df):
                row = df.iloc[source_idx]
                if "source" not in metadata:
                    metadata["source"] = row.get("source", "headlines")
                if "source_type" not in metadata:
                    metadata["source_type"] = row.get("source_type", "headlines")
                if "content_type" not in metadata:
                    metadata["content_type"] = "headline" if len(chunk["text"]) < 200 else "article"
    
    if not chunks:
        raise ValueError("No chunks created from dataset")
    
    logger.info(f"Created {len(chunks)} chunks from {len(df)} documents")
    
    # Step 3: Setup embeddings
    chunk_texts = [chunk["text"] for chunk in chunks]
    
    w2v_embedder = None
    if use_word2vec and ENHANCED_FEATURES_AVAILABLE:
        if w2v_model_path and os.path.exists(w2v_model_path):
            logger.info(f"Loading pre-trained Word2Vec model from {w2v_model_path}")
            w2v_embedder = Word2VecEmbedder()
            w2v_embedder.load(w2v_model_path)
        elif train_w2v:
            logger.info("Training Word2Vec model on dataset...")
            w2v_embedder = Word2VecEmbedder(vector_size=300)
            w2v_embedder.train(chunk_texts, epochs=w2v_epochs)
            
            # Save trained model
            w2v_output_path = "models/word2vec_model.model"
            os.makedirs(os.path.dirname(w2v_output_path), exist_ok=True)
            w2v_embedder.save(w2v_output_path)
            logger.info(f"Word2Vec model saved to {w2v_output_path}")
    
    # Step 4: Generate embeddings
    if use_hybrid and w2v_embedder and ENHANCED_FEATURES_AVAILABLE:
        logger.info("Using hybrid embedding approach (BGE + Word2Vec)...")
        embedder = HybridEmbedder(w2v_embedder=w2v_embedder)
        content_types = [chunk.get("metadata", {}).get("content_type", "auto") for chunk in chunks]
        embedding_result = embedder.embed_texts(chunk_texts, content_types=content_types)
        embeddings = embedding_result["embeddings"]
        strategies = embedding_result["strategies"]
        logger.info(f"Embedding strategies used: {set(strategies)}")
    else:
        logger.info("Using BGE embedding approach...")
        embedder = BGEEmbedder()
        embeddings = embedder.embed_texts(chunk_texts, batch_size=32)
    
    logger.info(f"Generated embeddings: shape {embeddings.shape}")
    
    # Step 5: Build FAISS index
    if use_multi_source and ENHANCED_FEATURES_AVAILABLE:
        logger.info("Building multi-source aware FAISS index...")
        vectorstore = MultiSourceFAISSStore(dimension=embeddings.shape[1])
        vectorstore.add_chunks(embeddings, chunks)
        
        # Display source statistics
        stats = vectorstore.get_source_statistics()
        logger.info(f"Source distribution: {stats}")
        
        # Save with sources
        sources_path = output_sources or "vectorstore/sources.pkl"
        vectorstore.save(output_index, output_chunks, sources_path)
        logger.info(f"Sources saved to {sources_path}")
    else:
        logger.info("Building standard FAISS index...")
        vectorstore = FAISSStore(dimension=embeddings.shape[1])
        vectorstore.add_chunks(embeddings, chunks)
        vectorstore.save(output_index, output_chunks)
    
    logger.info(f"Index built successfully! Total chunks: {len(chunks)}")
    logger.info(f"Index saved to {output_index}")
    logger.info(f"Chunks saved to {output_chunks}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build FAISS index from dataset. Supports enhanced features via optional flags."
    )
    parser.add_argument("--data-path", default="data/raw", help="Path to raw data")
    parser.add_argument("--output-index", default="vectorstore/faiss.index", help="Output index path")
    parser.add_argument("--output-chunks", default="vectorstore/chunks.pkl", help="Output chunks path")
    parser.add_argument("--output-sources", default=None, help="Output sources path (for multi-source)")
    
    # Enhanced features (optional)
    parser.add_argument("--use-word2vec", action="store_true", help="Use Word2Vec embeddings")
    parser.add_argument("--use-hybrid", action="store_true", help="Use hybrid embedding approach")
    parser.add_argument("--use-multi-source", action="store_true", help="Use multi-source aware storage")
    parser.add_argument("--w2v-model", type=str, help="Path to pre-trained Word2Vec model")
    parser.add_argument("--train-w2v", action="store_true", help="Train Word2Vec model on dataset")
    parser.add_argument("--w2v-epochs", type=int, default=10, help="Number of Word2Vec training epochs")
    
    args = parser.parse_args()
    
    build_index(
        data_path=args.data_path,
        output_index=args.output_index,
        output_chunks=args.output_chunks,
        output_sources=args.output_sources,
        use_word2vec=args.use_word2vec,
        use_hybrid=args.use_hybrid,
        use_multi_source=args.use_multi_source,
        w2v_model_path=args.w2v_model,
        train_w2v=args.train_w2v,
        w2v_epochs=args.w2v_epochs
    )

