"""
app.py — Streamlit UI for the SNN Research Assistant.

Run with:
    streamlit run app.py
"""

import os

import streamlit as st
from src.pipeline import ask

st.set_page_config(
    page_title="SNN Research Assistant",
    page_icon="🧠",
    layout="centered",
)

# --- Sidebar ---
with st.sidebar:
    st.title("About")
    st.markdown(
        """
        **SNN Research Assistant** is a RAG (Retrieval-Augmented Generation) pipeline
        built over academic papers on Spiking Neural Networks and neuromorphic computing.

        **Stack:**
        - LangChain + ChromaDB
        - HuggingFace sentence-transformers
        - Claude API (Anthropic)
        - Streamlit

        Ask questions grounded in the ingested papers — answers include source citations.
        """
    )


@st.cache_resource
def load_pipeline():
    """Warm up the embedding model and vectorstore once on app start."""
    from src.retriever import _get_vectorstore
    vs = _get_vectorstore()
    return vs._collection.count()


# --- Main UI ---
st.title("🧠 SNN Research Assistant")
st.caption("Ask questions about Spiking Neural Networks, grounded in your papers.")

if not os.getenv("ANTHROPIC_API_KEY"):
    st.error(
        "ANTHROPIC_API_KEY is not set. "
        "Add it as a Space secret (Settings → Variables and secrets)."
    )
    st.stop()

chunk_count = load_pipeline()

with st.sidebar:
    st.caption(f"DB: {chunk_count} chunks loaded")

with st.form("question_form"):
    question = st.text_input(
        "Your question",
        placeholder="e.g. What spiking neuron model does the thesis use?",
    )
    submitted = st.form_submit_button("Ask", type="primary")

if submitted and question.strip():
    with st.spinner("Retrieving context and generating answer..."):
        result = ask(question.strip())

    st.markdown("### Answer")
    st.write(result["answer"])

    if result["sources"]:
        st.markdown("### Sources")
        for source in result["sources"]:
            st.markdown(f"- `{source}`")
