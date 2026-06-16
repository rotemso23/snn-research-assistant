"""
Integration tests — require ChromaDB and embedding/reranking models.
No API key needed.

These tests verify that the retrieval pipeline components connect correctly:
ChromaDB is readable, the embedding model loads, and the CrossEncoder reranks.

Run with:
    pytest tests/test_integration.py -v
"""

import os
import pytest
from langchain_core.documents import Document
from src.retriever import retrieve_and_rerank, retrieve_fanout, _is_hebrew_dominant


def test_retrieval_returns_correct_count():
    """retrieve_and_rerank returns exactly top_k documents."""
    chunks = retrieve_and_rerank("What is the LIF neuron model?", fetch_k=20, top_k=5)
    assert len(chunks) == 5


def test_retrieval_documents_have_content():
    """Every returned document has non-empty page content."""
    chunks = retrieve_and_rerank("What is the LIF neuron model?", fetch_k=20, top_k=5)
    assert all(doc.page_content.strip() for doc in chunks)


def test_retrieval_documents_have_source_metadata():
    """Every returned document has a source field in its metadata."""
    chunks = retrieve_and_rerank("What is the LIF neuron model?", fetch_k=20, top_k=5)
    assert all(doc.metadata.get("source") for doc in chunks)


def test_retrieval_filters_hebrew_chunks():
    """No returned chunk should be Hebrew-dominant."""
    chunks = retrieve_and_rerank("What is the main goal of the thesis?", fetch_k=20, top_k=5)
    assert not any(_is_hebrew_dominant(doc.page_content) for doc in chunks)


def test_retrieval_thesis_boost_returns_thesis_chunks():
    """A thesis-specific query with force_thesis_boost should return at least one thesis chunk."""
    chunks = retrieve_and_rerank(
        "What models were proposed in the thesis?",
        fetch_k=20,
        top_k=5,
        force_thesis_boost=True,
    )
    sources = [doc.metadata.get("source", "") for doc in chunks]
    assert any("thesis" in s.lower() for s in sources)


def test_fanout_retrieval_no_duplicate_chunks():
    """Fan-out retrieval deduplicates — the same chunk should not appear twice."""
    sub_queries = [
        "What neural network models were proposed in the thesis?",
        "How were the proposed models evaluated and on what datasets?",
    ]
    chunks = retrieve_fanout(
        original_question="What models were proposed and how were they evaluated?",
        sub_queries=sub_queries,
        fetch_k=20,
        top_k=8,
    )
    contents = [doc.page_content[:200] for doc in chunks]
    assert len(contents) == len(set(contents)), "Duplicate chunks found in fanout result"


def test_first_pass_chunks_survive_retry():
    """Regression test: chunks graded as relevant in the first pass must survive the retry.

    Bug: grade_node finds good chunks but sets needs_more=True. retrieve_rerank_node
    fetches new chunks and returns only those — the original good chunks are lost.

    This test puts a known chunk in state["chunks"] (simulating what grade_node kept),
    runs retrieve_rerank_node in retry mode, and asserts that chunk is still in the output.

    Currently FAILS — will PASS after the fix is implemented.
    """
    from src.agent_graph import retrieve_rerank_node

    known_chunk = Document(
        page_content="The Leaky Integrate-and-Fire neuron accumulates input current and fires when threshold is crossed.",
        metadata={"source": "/papers/thesis_rotem_solomon.pdf"},
    )

    state = {
        "question":        "What is the LIF neuron model?",
        "k":               5,
        "use_hyde":        False,
        "multi_query":     False,
        "retrieval_query": "exponential domain computation neuromorphic hardware energy efficiency",
        "chunks":          [known_chunk],   # grade_node kept this from the first pass
        "answer":          "",
        "sources":         [],
        "need_more":       True,
        "retry_count":     1,              # this is a retry
        "query_type":      "simple_factual",
        "sub_queries":     [],
    }

    result = retrieve_rerank_node(state)

    result_contents = [doc.page_content for doc in result["chunks"]]
    assert known_chunk.page_content in result_contents, (
        "Chunk graded as relevant in first pass was lost during retry"
    )


@pytest.mark.e2e
def test_full_pipeline_smoke():
    """Full graph smoke test — one question in, one answer out. Requires API key."""
    from src.agent_graph import _agent_graph

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")

    result = _agent_graph.invoke({
        "question":        "What spiking neuron model does the thesis use?",
        "k":               5,
        "use_hyde":        False,
        "multi_query":     False,
        "retrieval_query": "What spiking neuron model does the thesis use?",
        "chunks":          [],
        "answer":          "",
        "sources":         [],
        "need_more":       False,
        "retry_count":     0,
        "query_type":      "",
        "sub_queries":     [],
    })

    assert result["answer"].strip(), "Answer should not be empty"
    assert isinstance(result["sources"], list)
    assert all(s.endswith(".pdf") for s in result["sources"])
