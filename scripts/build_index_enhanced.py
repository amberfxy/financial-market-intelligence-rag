"""Enhanced script to build FAISS index with hybrid embeddings and multi-source support."""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import logging
import argparse
from typing import List, Dict, Optional

from src.data.loader import load_kaggle_dataset, preprocess_data
from src.chunking.chunker import chunk_dataframe
from src.embeddings.embedder import BGEEmbedder
from src.embeddings.word2vec_embedder import Word2VecEmbedder
from src.embeddings.hybrid_embedder import HybridEmbedder
from src.vectorstore.faiss_store import FAISSStore
from src.vectorstore.multi_source_store import MultiSourceFAISSStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_enhanced_index(
    data_path: str = "data/raw",
    output_index: str = "vectorstore/faiss.index",
    output_chunks: str = "vectorstore/chunks.pkl",
    output_sources: Optional[str] = "vectorstore/sources.pkl",
    use_word2vec: bool = True,
    use_hybrid: bool = True,
    use_multi_source: bool = True,
    w2v_model_path: Optional[str] = None,
    train_w2v: bool = True,
    w2v_epochs: int = 10
):
    """
    Build FAISS index with enhanced features.
    
    Args:
        data_path: Path to raw data directory
        output_index: Output index file path
        output_chunks: Output chunks file path
        output_sources: Output sources file path (for multi-source)
        use_word2vec: Train/use Word2Vec embeddings
        use_hybrid: Use hybrid embedding approach
        use_multi_source: Use multi-source aware storage
        w2v_model_path: Path to pre-trained Word2Vec model (optional)
        train_w2v: Train Word2Vec model on dataset
        w2v_epochs: Number of epochs for Word2Vec training
    """
    logger.info("Starting enhanced index building process...")
    
    # Step 1: Load and preprocess data
    logger.info("Loading dataset...")
    df = load_kaggle_dataset(data_path)
    df = preprocess_data(df)
    
    # Add source metadata if not present
    if "source" not in df.columns and "source_type" not in df.columns:
        df["source_type"] = "headlines"  # Default source type
    
    # Step 2: Chunk documents with source metadata
    logger.info("Chunking documents...")
    chunks = chunk_dataframe(df, text_column="News Headline", max_tokens=250)
    
    # Add source metadata to chunks
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
                # Classify as headline for short texts
                metadata["content_type"] = "headline" if len(chunk["text"]) < 200 else "article"
    
    if not chunks:
        raise ValueError("No chunks created from dataset")
    
    logger.info(f"Created {len(chunks)} chunks from {len(df)} documents")
    
    # Step 3: Setup embeddings
    chunk_texts = [chunk["text"] for chunk in chunks]
    content_types = [chunk.get("metadata", {}).get("content_type", "auto") for chunk in chunks]
    
    w2v_embedder = None
    if use_word2vec:
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
    if use_hybrid and w2v_embedder:
        logger.info("Using hybrid embedding approach (BGE + Word2Vec)...")
        embedder = HybridEmbedder(w2v_embedder=w2v_embedder)
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
    if use_multi_source:
        logger.info("Building multi-source aware FAISS index...")
        vectorstore = MultiSourceFAISSStore(dimension=embeddings.shape[1])
        vectorstore.add_chunks(embeddings, chunks)
        
        # Display source statistics
        stats = vectorstore.get_source_statistics()
        logger.info(f"Source distribution: {stats}")
        
        # Save with sources
        vectorstore.save(output_index, output_chunks, output_sources or "vectorstore/sources.pkl")
    else:
        logger.info("Building standard FAISS index...")
        vectorstore = FAISSStore(dimension=embeddings.shape[1])
        vectorstore.add_chunks(embeddings, chunks)
        vectorstore.save(output_index, output_chunks)
    
    logger.info(f"Index built successfully! Total chunks: {len(chunks)}")
    logger.info(f"Index saved to {output_index}")
    logger.info(f"Chunks saved to {output_chunks}")
    
    if use_multi_source:
        logger.info(f"Sources saved to {output_sources or 'vectorstore/sources.pkl'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build enhanced FAISS index with hybrid embeddings and multi-source support"
    )
    parser.add_argument("--data-path", default="data/raw", help="Path to raw data")
    parser.add_argument("--output-index", default="vectorstore/faiss.index", help="Output index path")
    parser.add_argument("--output-chunks", default="vectorstore/chunks.pkl", help="Output chunks path")
    parser.add_argument("--output-sources", default="vectorstore/sources.pkl", help="Output sources path")
    parser.add_argument("--use-word2vec", action="store_true", default=True, help="Use Word2Vec embeddings")
    parser.add_argument("--no-word2vec", dest="use_word2vec", action="store_false", help="Disable Word2Vec")
    parser.add_argument("--use-hybrid", action="store_true", default=True, help="Use hybrid embedding approach")
    parser.add_argument("--no-hybrid", dest="use_hybrid", action="store_false", help="Disable hybrid embeddings")
    parser.add_argument("--use-multi-source", action="store_true", default=True, help="Use multi-source aware storage")
    parser.add_argument("--no-multi-source", dest="use_multi_source", action="store_false", help="Disable multi-source")
    parser.add_argument("--w2v-model", type=str, help="Path to pre-trained Word2Vec model")
    parser.add_argument("--train-w2v", action="store_true", default=True, help="Train Word2Vec model")
    parser.add_argument("--no-train-w2v", dest="train_w2v", action="store_false", help="Skip Word2Vec training")
    parser.add_argument("--w2v-epochs", type=int, default=10, help="Number of Word2Vec training epochs")
    
    args = parser.parse_args()
    
    build_enhanced_index(
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

