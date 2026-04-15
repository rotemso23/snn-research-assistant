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
    from src.retriever import _get_vectorstore, retrieve
    vs = _get_vectorstore()
    count = vs._collection.count()
    # Test 1: raw similarity_search — no MMR, no Hebrew filter
    raw_docs = vs.similarity_search("spiking neural network", k=5)
    # Test 2: full retrieve() — MMR + Hebrew filter
    filtered_docs = retrieve("spiking neural network", k=5)
    return count, len(raw_docs), len(filtered_docs)


# --- Main UI ---
st.title("🧠 SNN Research Assistant")
st.caption("Ask questions about Spiking Neural Networks, grounded in your papers.")

if not os.getenv("ANTHROPIC_API_KEY"):
    st.error(
        "ANTHROPIC_API_KEY is not set. "
        "Add it as a Space secret (Settings → Variables and secrets)."
    )
    st.stop()

chunk_count, raw_retrieved, filtered_retrieved = load_pipeline()

with st.sidebar:
    st.caption(f"DB: {chunk_count} chunks loaded")
    st.caption(f"Raw search: {raw_retrieved}/5 chunks")
    st.caption(f"After Hebrew filter: {filtered_retrieved}/5 chunks")

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
