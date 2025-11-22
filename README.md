# Financial Market Intelligence RAG System
### CS6120 Final Project  
**Team Members:** Soonbee Hwang & Xinyuan Fan (Amber)

---

## 1. Overview
This project implements a **Retrieval-Augmented Generation (RAG)** system for financial market intelligence.  
The system retrieves relevant financial news from a large corpus (20k+ Kaggle entries) and generates grounded answers using a **local LLM** with verifiable citations.

The goal is to provide accurate, up-to-date financial reasoning without hallucinations.

---

## 2. Project Architecture
Modules:

- **data/**: Kaggle dataset loading & preprocessing scripts  
- **src/**: ingestion, chunking, embedding, FAISS index, RAG logic  
- **app/**: Streamlit UI for querying the system  
- **docker/**: Dockerfile + docker-compose for deployment  
- **models/**: Instructions for downloading local LLM models  

---

## 3. Dataset

We use the Kaggle dataset:

**“Stock Market News Dataset” (20,383 financial news entries)**  
https://www.kaggle.com/datasets/aaron7sun/stocknews

This dataset provides:

- Clean and structured financial news  
- Date + headline + article body  
- High semantic quality  
- Satisfies the 10k+ requirement for CS6120

To download:

1. Install Kaggle CLI  
2. Place `kaggle.json` in `~/.kaggle/`  
3. Run:

```bash
cd data
kaggle datasets download -d aaron7sun/stocknews
unzip stocknews.zip
