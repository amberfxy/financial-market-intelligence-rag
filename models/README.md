# Model Downloads

## Local LLM: Mistral 7B GGUF

This project uses **Mistral 7B Instruct** in GGUF format for local inference.

### Download Instructions

1. **Recommended model**: `mistral-7b-instruct-v0.1.Q4_K_M.gguf` (~4.1 GB)
   - Good balance between quality and size
   - Download from: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF

2. **Alternative models** (choose based on your hardware):
   - `Q4_K_S.gguf` (~3.8 GB) - Smaller, faster
   - `Q5_K_M.gguf` (~4.9 GB) - Better quality
   - `Q8_0.gguf` (~7.0 GB) - Higher quality

### Download Command

```bash
# Using wget
cd models
wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf

# Or using huggingface-cli
pip install huggingface-hub
huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.1-GGUF mistral-7b-instruct-v0.1.Q4_K_M.gguf --local-dir models
```

### Update Model Path

After downloading, update the model path in:
- `src/rag/llm.py` (default path)
- Or set via environment variable

### Embedding Model

The **BGE-Large-en** model will be automatically downloaded on first use via `sentence-transformers`.
- Model: `BAAI/bge-large-en-v1.5`
- Size: ~1.3 GB
- Will be cached in `~/.cache/huggingface/`

