"""Semantic chunking using sentence-level logic."""

import tiktoken
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

# Initialize tokenizer for approximate token counting
try:
    encoding = tiktoken.get_encoding("cl100k_base")
except:
    encoding = None
    logger.warning("tiktoken not available, using character-based estimation")


def count_tokens(text: str) -> int:
    """Count approximate number of tokens in text."""
    if encoding:
        return len(encoding.encode(text))
    else:
        # Fallback: approximate 4 characters per token
        return len(text) // 4


def chunk_text(
    text: str,
    max_tokens: int = 250,
    overlap: int = 50,
    separator: str = "\n"
) -> List[Dict[str, any]]:
    """
    Split text into semantic chunks using sentence-level logic.
    
    Args:
        text: Input text to chunk
        max_tokens: Maximum tokens per chunk
        overlap: Number of tokens to overlap between chunks
        separator: Separator for splitting (default: newline for sentences)
        
    Returns:
        List of chunk dictionaries with 'text' and 'metadata'
    """
    if not text or len(text.strip()) == 0:
        return []
    
    # Split by sentences (newlines or periods)
    sentences = text.split(separator)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return [{"text": text, "metadata": {"chunk_id": 0}}]
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    chunk_id = 0
    
    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)
        
        # If single sentence exceeds max_tokens, split it further
        if sentence_tokens > max_tokens:
            # Save current chunk if exists
            if current_chunk:
                chunks.append({
                    "text": separator.join(current_chunk),
                    "metadata": {"chunk_id": chunk_id}
                })
                chunk_id += 1
                current_chunk = []
                current_tokens = 0
            
            # Split long sentence by words
            words = sentence.split()
            word_chunk = []
            word_tokens = 0
            
            for word in words:
                word_token_count = count_tokens(word)
                if word_tokens + word_token_count > max_tokens and word_chunk:
                    chunks.append({
                        "text": " ".join(word_chunk),
                        "metadata": {"chunk_id": chunk_id}
                    })
                    chunk_id += 1
                    word_chunk = [word]
                    word_tokens = word_token_count
                else:
                    word_chunk.append(word)
                    word_tokens += word_token_count
            
            if word_chunk:
                current_chunk = word_chunk
                current_tokens = word_tokens
        else:
            # Check if adding this sentence would exceed max_tokens
            if current_tokens + sentence_tokens > max_tokens and current_chunk:
                # Save current chunk
                chunks.append({
                    "text": separator.join(current_chunk),
                    "metadata": {"chunk_id": chunk_id}
                })
                chunk_id += 1
                
                # Start new chunk with overlap
                if overlap > 0 and len(current_chunk) > 0:
                    # Keep last few sentences for overlap
                    overlap_sentences = []
                    overlap_tokens = 0
                    for s in reversed(current_chunk):
                        s_tokens = count_tokens(s)
                        if overlap_tokens + s_tokens <= overlap:
                            overlap_sentences.insert(0, s)
                            overlap_tokens += s_tokens
                        else:
                            break
                    current_chunk = overlap_sentences
                    current_tokens = overlap_tokens
                else:
                    current_chunk = []
                    current_tokens = 0
            
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
    
    # Add final chunk
    if current_chunk:
        chunks.append({
            "text": separator.join(current_chunk),
            "metadata": {"chunk_id": chunk_id}
        })
    
    return chunks


def chunk_dataframe(
    df,
    text_column: str = "News Headline",
    max_tokens: int = 250,
    overlap: int = 50
) -> List[Dict[str, any]]:
    """
    Chunk all texts in a dataframe.
    
    Args:
        df: DataFrame with text data
        text_column: Name of column containing text
        max_tokens: Maximum tokens per chunk
        overlap: Overlap between chunks
        
    Returns:
        List of chunk dictionaries with text and metadata
    """
    all_chunks = []
    
    for idx, row in df.iterrows():
        text = str(row[text_column])
        chunks = chunk_text(text, max_tokens=max_tokens, overlap=overlap)
        
        for chunk in chunks:
            chunk["metadata"]["source_index"] = idx
            chunk["metadata"]["date"] = row.get("Date", "")
            chunk["metadata"]["headline"] = row.get("News Headline", "")[:100]
            all_chunks.append(chunk)
    
    logger.info(f"Created {len(all_chunks)} chunks from {len(df)} documents")
    return all_chunks

