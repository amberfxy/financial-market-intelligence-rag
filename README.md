# Financial Market Intelligence RAG System

**CS6120 Final Project**  
**Team Members:** Soonbee Hwang & Xinyuan Fan (Amber)  
**GitHub Repository:** https://github.com/amberfxy/financial-market-intelligence-rag

---

## 1. Overview

This project implements a **Retrieval-Augmented Generation (RAG)** system for financial market intelligence.  
The system retrieves relevant financial news from a large corpus (50k+ Kaggle entries) and generates grounded answers using a **local LLM** with verifiable citations.

The goal is to provide accurate, up-to-date financial reasoning without hallucinations.

---

## 2. Project Architecture

```
┌─────────────────┐
│  Streamlit UI   │
└────────┬────────┘
         │
┌────────▼────────┐
│   RAG Pipeline  │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
┌───▼───┐ ┌──▼────┐
│ FAISS │ │ Local │
│ Index │ │ LLM   │
└───┬───┘ └───────┘
    │
┌───▼──────────┐
│ BGE Embedder │
└─────────────┘
```

**Modules:**

- **data/**: Kaggle dataset loading & preprocessing scripts  
- **src/data/**: Data loading and preprocessing
- **src/chunking/**: Semantic chunking utilities
- **src/embeddings/**: BGE-Large-en embedding generation
- **src/vectorstore/**: FAISS IndexFlatL2 vector store
- **src/rag/**: RAG pipeline and local LLM inference
- **ui/**: Streamlit UI for querying the system  
- **scripts/**: Utility scripts for data processing
- **docker/**: Dockerfile + docker compose for deployment
- **models/**: Instructions for downloading local LLM models  

---

## 3. Dataset

We use the Kaggle dataset:

**"Daily News for Stock Market Prediction" (50,000+ financial news headlines)**  
https://www.kaggle.com/datasets/aaron7sun/stocknews

This dataset provides:

- Clean and structured financial news  
- Date + headline + article body  
- High semantic quality  
- Satisfies the 10k+ requirement for CS6120

### Download Instructions:

1. Install Kaggle CLI: `pip install kaggle`
2. Get your API credentials from https://www.kaggle.com/account
3. Place `kaggle.json` in `~/.kaggle/`
4. Run:

```bash
./scripts/download_data.sh
```

Or manually:

```bash
cd data/raw
kaggle datasets download -d aaron7sun/stocknews
unzip stocknews.zip
```

---

## 4. Setup and Installation

### Prerequisites

- Python 3.10+
- CUDA (optional, for GPU acceleration)
- Docker (for containerized deployment)

### Local Setup

1. **Clone the repository:**
```bash
git clone https://github.com/amberfxy/financial-market-intelligence-rag.git
cd financial-market-intelligence-rag
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Download the dataset** (see Dataset section above)

4. **Download the LLM model:**
```bash
cd models
# See models/README.md for download instructions
wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf
```

5. **Build the FAISS index:**
```bash
python scripts/build_index.py
```

6. **Run the Streamlit app:**
```bash
streamlit run ui/app.py
```

The app will be available at `http://localhost:8501`

---

## 5. Docker Deployment

### Build and Run with Docker

1. **Build the Docker image:**
```bash
docker build -t financial-rag-system .
```

2. **Run with docker compose:**
```bash
docker compose up -d
```

3. **Access the app:**
   - Local: http://localhost:8501
   - GCP VM: http://<VM_IP>:8501

### Docker Features

- Automatic dependency installation
- Health checks
- Volume mounts for data and models
- Port mapping for external access

---

## 6. Usage

### Web Interface

1. Open the Streamlit app in your browser
2. Click "Initialize System" in the sidebar
3. Enter your question in the query box
4. Click "Search" to get an answer with citations

### Example Queries

- "Why did NVDA stock fall after earnings?"
- "What were the main market trends in 2015?"
- "How did the financial crisis affect tech stocks?"

### Features

- Real-time inference with local LLM
- Clickable citations to source documents
- Adjustable retrieval parameters (Top-K)
- Latency measurement and display
- Expandable evidence view

---

## 7. Technical Details

### Components

- **Embedding Model**: BGE-Large-en-v1.5 (1024 dimensions)
- **Vector Store**: FAISS IndexFlatL2 (exact L2 distance)
- **LLM**: Mistral 7B Instruct GGUF (local inference)
- **Chunking**: Sentence-level semantic chunking (~250 tokens)
- **Retrieval**: Top-K similarity search with cosine similarity

### Performance Targets

- Retrieval latency: <50ms
- End-to-end latency: <1.5 seconds
- Citation accuracy: Verified against source documents

---

## 8. Project Requirements Compliance

- **Frontend**: Streamlit-based UI  
- **Database**: 50k+ entries (exceeds 10k requirement)  
- **Local LLM**: Mistral 7B GGUF via llama-cpp-python  
- **Citations**: Clickable references to source documents  
- **Dockerization**: Complete Docker setup  
- **Real-time Inference**: Live query processing  

---

## 9. File Structure

```
.
├── data/
│   ├── raw/              # Raw Kaggle dataset
│   └── processed/        # Processed data
├── src/
│   ├── data/             # Data loading
│   ├── chunking/         # Text chunking
│   ├── embeddings/       # BGE embeddings
│   ├── vectorstore/      # FAISS store
│   └── rag/              # RAG pipeline
├── ui/
│   └── app.py            # Streamlit app
├── scripts/
│   ├── build_index.py    # Index building script
│   └── download_data.sh  # Data download script
├── models/               # LLM models
├── vectorstore/          # FAISS index files
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 10. Troubleshooting

### Model Not Found
- Ensure Mistral model is downloaded to `models/` directory
- Check model path in `src/rag/llm.py`

### Index Not Found
- Run `python scripts/build_index.py` to build the index
- Ensure data is downloaded and processed

### CUDA/GPU Issues
- For CPU-only: llama-cpp-python will use CPU automatically
- For GPU: Install CUDA-enabled version (see Dockerfile comments)

---

## 11. Contributors

- **Soonbee Hwang** (hwang.soon@northeastern.edu)
- **Xinyuan Fan (Amber)** (fan.xinyua@northeastern.edu)

---

## 12. License

This project is for CS6120 course purposes.

---

## 13. References

- Dataset: https://www.kaggle.com/datasets/aaron7sun/stocknews
- BGE Model: https://huggingface.co/BAAI/bge-large-en-v1.5
- Mistral Model: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF
- FAISS: https://github.com/facebookresearch/faiss
