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
        **SNN Research Assistant** is an agentic RAG (Retrieval-Augmented Generation) pipeline
        built over academic papers on Spiking Neural Networks and neuromorphic computing.

        **Stack:**
        - LangGraph + LangChain + ChromaDB
        - HuggingFace (BGE embeddings + CrossEncoder reranker)
        - Claude API (Anthropic)
        - Streamlit

        Ask a question — the assistant searches the papers and returns a cited answer.
        """
    )


@st.cache_resource
def load_pipeline():
    """Warm up the embedding model and vectorstore once on app start."""
    from src.retriever import _get_vectorstore
    _get_vectorstore()


# --- Main UI ---
st.title("🧠 SNN Research Assistant")
st.caption("Powered by agentic RAG — ask anything about SNNs and neuromorphic computing.")
st.info(
    "⏳ **Heads up:** this Space runs on CPU, so a full query — including retrieval and reranking — "
    "typically takes **20–40 seconds**. Thanks for your patience!",
    icon=None,
)

if not os.getenv("ANTHROPIC_API_KEY"):
    st.error(
        "ANTHROPIC_API_KEY is not set. "
        "Add it as a Space secret (Settings → Variables and secrets)."
    )
    st.stop()

load_pipeline()

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

    with st.expander("🔍 Pipeline internals"):
        col1, col2 = st.columns(2)
        col1.metric("Route", result.get("query_type") or "—")
        col2.metric("Attempts", f"{result.get('retry_count', 0) + 1} / 3")
        sub_queries = result.get("sub_queries", [])
        if sub_queries:
            st.markdown("**Sub-queries**")
            for i, q in enumerate(sub_queries, 1):
                st.markdown(f"{i}. {q}")

    if result["sources"]:
        st.markdown("### Sources")
        for source in result["sources"]:
            st.markdown(f"- `{source}`")
