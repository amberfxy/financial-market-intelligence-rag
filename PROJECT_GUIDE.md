# Financial Market Intelligence RAG System - Project Guide

## Overview

A **Retrieval-Augmented Generation (RAG) system** for financial market Q&A.

**Workflow**: User Query → Retrieve Documents → Generate Answer with Citations

### Core Components

- **Data**: 50,000+ financial news articles (Kaggle dataset)
- **Embedding Model**: BGE-Large-en-v1.5 (1024-dim vectors)
- **Vector Database**: FAISS IndexFlatL2 (fast similarity search)
- **LLM**: Mistral 7B GGUF (local inference)
- **UI**: Streamlit web application

---

## System Architecture

```
User Query
    ↓
Streamlit UI (ui/app.py)
    ↓
RAG Pipeline (src/rag/pipeline.py)
    ↓
    ├── Embed Query (src/embeddings/embedder.py)
    ├── Retrieve Top-K (src/vectorstore/faiss_store.py)
    └── Generate Answer (src/rag/llm.py)
    ↓
Return Answer + Citations
```

---

## File Structure

### Root Files

- **`README.md`**: Main documentation, quick start guide
- **`requirements.txt`**: Python dependencies
- **`Dockerfile`**: Docker image configuration
- **`Financial_Market_RAG_Colab.ipynb`**: Colab version for cloud execution

### `src/` - Source Code

#### `src/data/loader.py`
- Loads Kaggle dataset from `data/raw/`
- Preprocesses data (deduplication, HTML removal, normalization)
- **Used by**: `scripts/build_index.py`

#### `src/chunking/chunker.py`
- Splits text into semantic chunks (~250 tokens, 50 token overlap)
- Sentence-level chunking for semantic integrity
- **Used by**: `scripts/build_index.py`

#### `src/embeddings/embedder.py`
- Generates embeddings using BGE-Large-en-v1.5
- Batch processing with GPU/CPU auto-detection
- **Used by**: `scripts/build_index.py`, `src/rag/pipeline.py`

#### `src/vectorstore/faiss_store.py`
- FAISS vector database wrapper
- Stores embeddings and chunks
- Fast Top-K similarity search
- **Used by**: `scripts/build_index.py`, `src/rag/pipeline.py`

#### `src/rag/llm.py`
- Local LLM inference (Mistral 7B GGUF)
- Generates answers with citation extraction
- **Used by**: `src/rag/pipeline.py`

#### `src/rag/pipeline.py`
- **Core RAG orchestrator**
- Coordinates: embedding → retrieval → generation
- Main method: `query()` - processes user queries
- **Used by**: `ui/app.py`

### `ui/app.py`
- Streamlit web application
- Query interface and result display
- System initialization and caching
- **Start**: `streamlit run ui/app.py`

### `scripts/`

#### `scripts/build_index.py`
- **Main script for building FAISS index**
- Workflow: Load → Chunk → Embed → Build Index → Save
- **Run**: `python scripts/build_index.py`
- **Output**: `vectorstore/faiss.index`, `vectorstore/chunks.pkl`

#### `scripts/download_data.sh`
- Downloads Kaggle dataset using API
- Extracts to `data/raw/`

---

## Data Flow

### 1. Index Building (One-time)

```
Raw Data (data/raw/*.csv)
    ↓ loader.py
Cleaned DataFrame
    ↓ chunker.py
Text Chunks (80k-100k chunks)
    ↓ embedder.py
Embeddings (n_chunks × 1024)
    ↓ faiss_store.py
FAISS Index + Chunks
    ↓ save
vectorstore/faiss.index + chunks.pkl
```

### 2. Query Processing (Real-time)

```
User Query
    ↓ embedder.py
Query Embedding (1024-dim)
    ↓ faiss_store.py
Top-K Similar Chunks
    ↓ llm.py
Answer with Citations
    ↓
Return Results
```

---

## How to Run

### Method 1: Local (Development)

```bash
# 1. Clone repository
git clone https://github.com/amberfxy/financial-market-intelligence-rag.git
cd financial-market-intelligence-rag

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download data (Kaggle API or manual)
# Place CSV files in data/raw/

# 4. Download model (~4.1GB)
mkdir -p models
cd models
wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf
cd ..

# 5. Build index
python scripts/build_index.py

# 6. Run application
streamlit run ui/app.py
```

### Method 2: Docker (Deployment)

```bash
# 1. Build image
docker build -t financial-rag-system .

# 2. Ensure data/model/index exist

# 3. Run container
docker-compose up -d
# Or: docker run -p 8501:8501 -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models -v $(pwd)/vectorstore:/app/vectorstore financial-rag-system

# 4. Access: http://localhost:8501
```

### Method 3: Google Colab (Demo)

1. Upload `Financial_Market_RAG_Colab.ipynb` to Colab
2. Enable GPU runtime
3. Run cells in order:
   - Install dependencies
   - Configure Kaggle API token
   - Download dataset
   - Download model
   - Clone code
   - Build index
   - Run Streamlit app

---

## Key Files Summary

| File | Purpose | When Used |
|------|---------|-----------|
| `ui/app.py` | Web interface | Every run |
| `src/rag/pipeline.py` | RAG core logic | Every query |
| `src/embeddings/embedder.py` | Generate embeddings | Index build + queries |
| `src/vectorstore/faiss_store.py` | Vector search | Index build + queries |
| `src/rag/llm.py` | LLM inference | Every query |
| `scripts/build_index.py` | Build index | One-time (when data updates) |

---

## Dependencies

```
ui/app.py
    ↓ imports
src/rag/pipeline.py
    ↓ imports
    ├── src/embeddings/embedder.py
    ├── src/vectorstore/faiss_store.py
    └── src/rag/llm.py

scripts/build_index.py
    ↓ imports
    ├── src/data/loader.py
    ├── src/chunking/chunker.py
    ├── src/embeddings/embedder.py
    └── src/vectorstore/faiss_store.py
```

---

## FAQ

**Q: How long does index building take?**  
A: CPU: 30-60 min, GPU: 2-5 min (for 50k documents)

**Q: What's the query response time?**  
A: Target: <1.5s, Actual: 1.2-1.4s (CPU), 0.3-0.5s (GPU)

**Q: How to update data?**  
A: Replace CSV files in `data/raw/`, then run `python scripts/build_index.py`

---


