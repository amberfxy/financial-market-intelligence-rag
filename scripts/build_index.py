"""Script to build FAISS index from processed data."""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import logging
from src.data.loader import load_kaggle_dataset, preprocess_data
from src.chunking.chunker import chunk_dataframe
from src.embeddings.embedder import BGEEmbedder
from src.vectorstore.faiss_store import FAISSStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_index(
    data_path: str = "data/raw",
    output_index: str = "vectorstore/faiss.index",
    output_chunks: str = "vectorstore/chunks.pkl"
):
    """Build FAISS index from dataset."""
    logger.info("Starting index building process...")
    
    # Step 1: Load and preprocess data
    logger.info("Loading dataset...")
    df = load_kaggle_dataset(data_path)
    df = preprocess_data(df)
    
    # Step 2: Chunk documents
    logger.info("Chunking documents...")
    chunks = chunk_dataframe(df, text_column="News Headline", max_tokens=250)
    
    if not chunks:
        raise ValueError("No chunks created from dataset")
    
    # Step 3: Generate embeddings
    logger.info("Generating embeddings...")
    embedder = BGEEmbedder()
    
    chunk_texts = [chunk["text"] for chunk in chunks]
    embeddings = embedder.embed_texts(chunk_texts, batch_size=32)
    
    # Step 4: Build FAISS index
    logger.info("Building FAISS index...")
    vectorstore = FAISSStore(dimension=embeddings.shape[1])
    vectorstore.add_chunks(embeddings, chunks)
    
    # Step 5: Save index
    logger.info("Saving index...")
    vectorstore.save(output_index, output_chunks)
    
    logger.info(f"Index built successfully! Total chunks: {len(chunks)}")
    logger.info(f"Index saved to {output_index}")
    logger.info(f"Chunks saved to {output_chunks}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build FAISS index from dataset")
    parser.add_argument("--data-path", default="data/raw", help="Path to raw data")
    parser.add_argument("--output-index", default="vectorstore/faiss.index", help="Output index path")
    parser.add_argument("--output-chunks", default="vectorstore/chunks.pkl", help="Output chunks path")
    
    args = parser.parse_args()
    
    build_index(args.data_path, args.output_index, args.output_chunks)

