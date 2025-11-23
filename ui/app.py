"""Streamlit UI for Financial Market Intelligence RAG System."""

import streamlit as st
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings.embedder import BGEEmbedder
from src.vectorstore.faiss_store import FAISSStore
from src.rag.llm import LocalLLM
from src.rag.pipeline import RAGPipeline
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Financial Market Intelligence RAG",
    page_icon="📈",
    layout="wide"
)

# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
    st.session_state.initialized = False


@st.cache_resource
def load_pipeline():
    """Load RAG pipeline components (cached)."""
    try:
        # Load components
        embedder = BGEEmbedder()
        vectorstore = FAISSStore()
        vectorstore.load()
        
        # Try to load LLM (may fail if model not downloaded)
        try:
            llm = LocalLLM()
        except FileNotFoundError:
            st.warning("LLM model not found. Please download the model first.")
            llm = None
        
        if llm:
            pipeline = RAGPipeline(embedder, vectorstore, llm, top_k=5)
            return pipeline
        else:
            return None
    except Exception as e:
        st.error(f"Error loading pipeline: {str(e)}")
        return None


def main():
    """Main Streamlit app."""
    st.title("📈 Financial Market Intelligence RAG System")
    st.markdown("**CS6120 Final Project** | Soonbee Hwang & Xinyuan Fan (Amber)")
    st.markdown("---")
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Load pipeline
        if st.button("🔄 Initialize System", type="primary"):
            with st.spinner("Loading models and vector store..."):
                pipeline = load_pipeline()
                if pipeline:
                    st.session_state.pipeline = pipeline
                    st.session_state.initialized = True
                    st.success("System initialized successfully!")
                else:
                    st.error("Failed to initialize system. Check logs.")
        
        if st.session_state.initialized:
            st.success("✅ System Ready")
            
            # Adjustable parameters
            st.subheader("Retrieval Parameters")
            top_k = st.slider("Top-K Results", 1, 10, 5)
            
            if st.session_state.pipeline:
                st.session_state.pipeline.top_k = top_k
        else:
            st.info("Click 'Initialize System' to start")
    
    # Main content area
    if not st.session_state.initialized or st.session_state.pipeline is None:
        st.info("👈 Please initialize the system from the sidebar first.")
        st.markdown("""
        ### Setup Instructions:
        1. Download the Kaggle dataset to `data/raw/`
        2. Run the data processing script to create embeddings
        3. Download the Mistral 7B GGUF model to `models/`
        4. Click 'Initialize System' in the sidebar
        """)
        return
    
    # Query input
    st.subheader("💬 Ask a Question")
    query = st.text_input(
        "Enter your question about financial markets:",
        placeholder="e.g., Why did NVDA stock fall after earnings?",
        key="query_input"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        submit_button = st.button("🔍 Search", type="primary", use_container_width=True)
    
    # Process query
    if submit_button and query:
        with st.spinner("Processing query..."):
            result = st.session_state.pipeline.query(query, top_k=top_k)
        
        # Display answer
        st.markdown("---")
        st.subheader("📝 Answer")
        st.write(result["answer"])
        
        # Display citations
        if result.get("citations"):
            st.markdown("---")
            st.subheader("📚 Citations")
            
            with st.expander(f"View {len(result['citations'])} citation(s)", expanded=False):
                for citation in result["citations"]:
                    with st.container():
                        st.markdown(f"**[{citation['citation_id']}]**")
                        metadata = citation.get("metadata", {})
                        if metadata.get("headline"):
                            st.caption(f"Source: {metadata['headline']}")
                        if metadata.get("date"):
                            st.caption(f"Date: {metadata['date']}")
                        st.text(citation["chunk"]["text"][:500] + "...")
                        st.markdown("---")
        
        # Display retrieved chunks
        if result.get("retrieved_chunks"):
            st.markdown("---")
            st.subheader("🔍 Retrieved Evidence")
            
            with st.expander(f"View {len(result['retrieved_chunks'])} retrieved chunk(s)", expanded=False):
                for i, chunk_result in enumerate(result["retrieved_chunks"], 1):
                    with st.container():
                        st.markdown(f"**Chunk {i}** (Score: {chunk_result['score']:.4f})")
                        metadata = chunk_result.get("metadata", {})
                        if metadata.get("headline"):
                            st.caption(f"Source: {metadata['headline']}")
                        st.text(chunk_result["chunk"]["text"][:300] + "...")
                        st.markdown("---")
        
        # Display latency
        if result.get("latency"):
            st.markdown("---")
            st.subheader("⏱️ Performance Metrics")
            latency = result["latency"]
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total", f"{latency['total']:.2f}s")
            with col2:
                st.metric("Embedding", f"{latency['embedding']:.2f}s")
            with col3:
                st.metric("Retrieval", f"{latency['retrieval']:.2f}s")
            with col4:
                st.metric("Generation", f"{latency['generation']:.2f}s")
            
            # Check if latency target is met
            if latency['total'] < 1.5:
                st.success(f"✅ Latency target met (<1.5s): {latency['total']:.2f}s")
            else:
                st.warning(f"⚠️ Latency exceeds target: {latency['total']:.2f}s")


if __name__ == "__main__":
    main()

