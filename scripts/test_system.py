"""Test script to verify system components."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all modules can be imported."""
    logger.info("Testing imports...")
    try:
        from src.data.loader import load_kaggle_dataset, preprocess_data
        from src.chunking.chunker import chunk_text
        from src.embeddings.embedder import BGEEmbedder
        from src.vectorstore.faiss_store import FAISSStore
        from src.rag.llm import LocalLLM
        from src.rag.pipeline import RAGPipeline
        logger.info("✅ All imports successful")
        return True
    except Exception as e:
        logger.error(f"❌ Import failed: {e}")
        return False


def test_embedder():
    """Test embedding generation."""
    logger.info("Testing embedder...")
    try:
        embedder = BGEEmbedder()
        test_text = "This is a test sentence."
        embedding = embedder.embed_texts([test_text])
        assert embedding.shape[0] == 1
        assert embedding.shape[1] == 1024  # BGE-Large-en dimension
        logger.info("✅ Embedder test passed")
        return True
    except Exception as e:
        logger.error(f"❌ Embedder test failed: {e}")
        return False


def test_chunker():
    """Test text chunking."""
    logger.info("Testing chunker...")
    try:
        from src.chunking.chunker import chunk_text
        test_text = "This is sentence one. This is sentence two. " * 50
        chunks = chunk_text(test_text, max_tokens=50)
        assert len(chunks) > 0
        logger.info(f"✅ Chunker test passed (created {len(chunks)} chunks)")
        return True
    except Exception as e:
        logger.error(f"❌ Chunker test failed: {e}")
        return False


def test_vectorstore():
    """Test FAISS vector store."""
    logger.info("Testing vector store...")
    try:
        from src.vectorstore.faiss_store import FAISSStore
        import numpy as np
        
        store = FAISSStore(dimension=1024)
        test_embeddings = np.random.rand(5, 1024).astype('float32')
        test_chunks = [{"text": f"Chunk {i}", "metadata": {}} for i in range(5)]
        store.add_chunks(test_embeddings, test_chunks)
        
        query = np.random.rand(1024).astype('float32')
        results = store.search(query, k=3)
        assert len(results) == 3
        logger.info("✅ Vector store test passed")
        return True
    except Exception as e:
        logger.error(f"❌ Vector store test failed: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("=" * 50)
    logger.info("Running system tests...")
    logger.info("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Chunker", test_chunker),
        ("Embedder", test_embedder),
        ("Vector Store", test_vectorstore),
    ]
    
    results = []
    for name, test_func in tests:
        logger.info(f"\n--- Testing {name} ---")
        result = test_func()
        results.append((name, result))
    
    logger.info("\n" + "=" * 50)
    logger.info("Test Results:")
    logger.info("=" * 50)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{name}: {status}")
    
    all_passed = all(result for _, result in results)
    if all_passed:
        logger.info("\n🎉 All tests passed!")
    else:
        logger.info("\n⚠️  Some tests failed. Please check the errors above.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

