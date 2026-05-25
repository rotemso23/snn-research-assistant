"""
graph.py — LangGraph StateGraph for the SNN RAG pipeline.

Graph topology:
    START → [retrieve_rerank] → [generate_answer] ──┬──► END
                                                     │
                                        (needs_fallback)
                                                     ▼
                                           [fallback_retrieve] → [generate_answer] → END

Nodes call the existing retrieve_and_rerank / retrieve / generate functions
from src/retriever.py and src/generator.py — those modules are unchanged.
"""

from typing_extensions import TypedDict
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END

from src.retriever import retrieve_and_rerank, retrieve
from src.generator import generate


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class RAGState(TypedDict):
    # Inputs — set at graph entry, never mutated by nodes
    question: str
    k: int
    use_hyde: bool
    multi_query: bool
    # Set by retrieve_rerank; overwritten by fallback_retrieve if fallback runs
    chunks: list[Document]
    # Set by generate_answer
    answer: str
    sources: list[str]
    # Circuit-breaker: prevents infinite loop if fallback answer is also insufficient
    fallback_attempted: bool


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def retrieve_rerank(state: RAGState) -> dict:
    """Node 1: full retrieval + reranking pipeline (HyDE, multi-query, CrossEncoder)."""
    chunks = retrieve_and_rerank(
        state["question"],
        fetch_k=20,
        top_k=state["k"],
        use_hyde=state["use_hyde"],
        multi_query=state["multi_query"],
    )
    return {"chunks": chunks}


def generate_answer(state: RAGState) -> dict:
    """Node 2: call Claude with the current chunks and return answer + sources."""
    result = generate(state["question"], state["chunks"])
    return {"answer": result["answer"], "sources": result["sources"]}


def fallback_retrieve(state: RAGState) -> dict:
    """Node 3: augmented plain-MMR retrieval for the fallback path.

    Appends domain terms to the query so the embedding shifts toward structural
    thesis chunks that HyDE + CrossEncoder may have missed.
    Sets fallback_attempted=True so route_after_generate terminates on the next pass.
    """
    augmented = state["question"] + " spiking neural network thesis"
    chunks = retrieve(augmented, k=15)
    return {"chunks": chunks, "fallback_attempted": True}


# ---------------------------------------------------------------------------
# Routing function (conditional edge after generate_answer)
# ---------------------------------------------------------------------------

def route_after_generate(state: RAGState) -> str:
    """Decide whether to invoke the fallback path or stop.

    Returns:
        "fallback_retrieve" if the answer signals insufficient context
        and the fallback has not yet been attempted.
        END otherwise — including when the fallback already ran, so the
        graph terminates even if the fallback answer is also "I don't know".
    """
    insufficient = "does not contain enough information" in state["answer"]
    already_tried = state.get("fallback_attempted", False)
    if insufficient and not already_tried:
        return "fallback_retrieve"
    return END


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph():
    """Build and compile the RAG StateGraph."""
    builder = StateGraph(RAGState)

    builder.add_node("retrieve_rerank", retrieve_rerank)
    builder.add_node("generate_answer", generate_answer)
    builder.add_node("fallback_retrieve", fallback_retrieve)

    builder.add_edge(START, "retrieve_rerank")
    builder.add_edge("retrieve_rerank", "generate_answer")

    builder.add_conditional_edges(
        "generate_answer",
        route_after_generate,
        {"fallback_retrieve": "fallback_retrieve", END: END},
    )

    builder.add_edge("fallback_retrieve", "generate_answer")

    return builder.compile()


# Module-level singleton — graph is built once at import, not on every ask() call
_graph = build_graph()
