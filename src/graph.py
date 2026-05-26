"""
graph.py — LangGraph StateGraph for the SNN RAG pipeline.

Graph topology:
    START
      │
      ▼
    [retrieve_rerank]
      │
      ▼
    [grade_documents] ──── all_filtered=False ───────────────────────► [generate_answer]
      │                                                                         │
      └── all_filtered=True + not rewritten → [rewrite_query]        route_after_generate
                                   │                                    ├── good → END
                                   ▼                                    └── "I don't know"
                            [retrieve_rerank]                                   ▼
                                   │                                   [fallback_retrieve]
                                   ▼                                           │
                            [grade_documents]                                  ▼
                                   │                                   [generate_answer] → END
                                   └── (always → generate_answer)

Nodes call the existing retrieve_and_rerank / retrieve / generate functions
from src/retriever.py and src/generator.py — those modules are unchanged.
"""

import anthropic
from typing_extensions import TypedDict
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END

from src.retriever import retrieve_and_rerank, retrieve
from src.generator import generate

# Single client instance reused across all node calls
_client = anthropic.Anthropic()


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class RAGState(TypedDict):
    # Inputs — set at graph entry, never mutated by nodes
    question: str
    k: int
    use_hyde: bool
    multi_query: bool
    # Starts equal to question; overwritten by rewrite_query if triggered
    retrieval_query: str
    # Set by retrieve_rerank; filtered by grade_documents; overwritten by fallback_retrieve
    chunks: list[Document]
    # Set by generate_answer
    answer: str
    sources: list[str]
    # Circuit-breaker: prevents infinite fallback loop
    fallback_attempted: bool
    # Circuit-breaker: prevents infinite rewrite loop (fires at most once)
    rewrite_attempted: bool
    # Set by grade_documents: True when the grader rejected all chunks before
    # the safety floor restored them. Used by route_after_grading to trigger
    # rewrite_query even though state["chunks"] is never empty.
    all_filtered: bool


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def retrieve_rerank(state: RAGState) -> dict:
    """Node 1: full retrieval + reranking pipeline (HyDE, multi-query, CrossEncoder).

    Uses retrieval_query (not question directly) so that rewrite_query can
    substitute a better search query without changing the original question.
    HyDE is skipped on the second pass (after rewrite_query) — the rewritten
    query is already a sharpened technical string; wrapping it in a hypothetical
    answer adds noise rather than signal.
    """
    chunks = retrieve_and_rerank(
        state["retrieval_query"],
        fetch_k=20,
        top_k=state["k"],
        use_hyde=state["use_hyde"] and not state.get("rewrite_attempted", False),
        multi_query=state["multi_query"],
    )
    return {"chunks": chunks}


def grade_documents(state: RAGState) -> dict:
    """Node 2: filter retrieved chunks by relevance using Claude Haiku.

    Calls Haiku once for all chunks in a single request (simple binary
    classification — no need for Sonnet). Grades against the original
    question (ground truth of what the user needs), not the retrieval_query.

    Grading is intentionally lenient: a chunk passes if it contains information
    that could help answer *any part* of the question, even partially or
    indirectly. Only chunks that are completely off-topic are dropped.

    Safety floor: if the grader rejects every chunk (e.g. abstract questions
    where relevant chunks don't explicitly mention the topic keyword), all
    original chunks are kept so the generator always has context to work with.
    """
    if not state["chunks"]:
        return {"chunks": [], "all_filtered": False}

    chunks_text = "\n\n".join(
        f"Chunk {i+1}:\n{doc.page_content[:400]}"
        for i, doc in enumerate(state["chunks"])
    )
    response = _client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=50,
        system=(
            "You are grading retrieved document chunks for a RAG pipeline on "
            "spiking neural networks (SNNs) and neuromorphic computing. "
            "Say 'yes' if a chunk contains information that could help answer "
            "ANY part of the question — even partially or indirectly. "
            "Say 'no' ONLY if the chunk is completely unrelated to the question topic. "
            "When in doubt, say 'yes'. "
            f"There are {len(state['chunks'])} chunks. "
            "Output exactly 'yes' or 'no' on its own line for each chunk — "
            "no numbering, no extra text."
        ),
        messages=[{"role": "user", "content": f"Question: {state['question']}\n\n{chunks_text}"}],
    )
    grades = [
        line.strip().lower()
        for line in response.content[0].text.strip().split("\n")
        if line.strip()
    ]
    if len(grades) != len(state["chunks"]):
        # Haiku returned an unexpected number of grade lines — use whatever grades
        # we got, and keep the remaining ungraded chunks (benefit of the doubt).
        graded_relevant = [
            doc for doc, grade in zip(state["chunks"], grades)
            if grade == "yes"
        ]
        ungraded = state["chunks"][len(grades):]
        relevant = graded_relevant + ungraded
    else:
        relevant = [
            doc for doc, grade in zip(state["chunks"], grades)
            if grade == "yes"
        ]
    # Safety floor: never hand the generator zero chunks.
    # Record all_filtered=True *before* restoring originals so route_after_grading
    # can still detect poor retrieval quality and trigger rewrite_query.
    all_filtered = not relevant
    if all_filtered:
        relevant = state["chunks"]
    return {"chunks": relevant, "all_filtered": all_filtered}


def rewrite_query(state: RAGState) -> dict:
    """Node 3: rewrite the question into a better academic search query.

    Uses Claude Haiku to produce a query rich in technical vocabulary likely
    to appear in research papers on SNNs and neuromorphic computing.
    Passes the failed chunks to the rewriter so it can steer away from topics
    that were already retrieved and deemed irrelevant.
    Sets rewrite_attempted=True (circuit-breaker) so this node fires at most once.
    """
    failed = "\n".join(
        f"- {doc.page_content[:200]}" for doc in state["chunks"]
    )
    response = _client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        system=(
            "You are a search query optimizer for academic paper retrieval on "
            "spiking neural networks and neuromorphic computing. "
            "Rewrite the given question into a more effective search query using "
            "technical vocabulary likely to appear in research papers. "
            "Output only the rewritten query, nothing else."
        ),
        messages=[{
            "role": "user",
            "content": (
                f"Question: {state['question']}\n\n"
                f"The following chunks were retrieved but were all irrelevant:\n{failed}\n\n"
                "Rewrite the question into a more precise search query that retrieves "
                "chunks more directly relevant to answering it than the ones above."
            ),
        }],
    )
    rewritten = response.content[0].text.strip()
    return {"retrieval_query": rewritten, "rewrite_attempted": True}


def generate_answer(state: RAGState) -> dict:
    """Node 4: call Claude with the current chunks and return answer + sources."""
    result = generate(state["question"], state["chunks"])
    return {"answer": result["answer"], "sources": result["sources"]}


def fallback_retrieve(state: RAGState) -> dict:
    """Node 5: plain-MMR retrieval for the fallback path.

    Skips HyDE, multi-query, and CrossEncoder — tries a genuinely different
    retrieval strategy in case those steps biased or narrowed the candidate pool.
    Fetches a wider k=15 to cast a broader net.
    Sets fallback_attempted=True so route_after_generate terminates on the next pass.
    """
    chunks = retrieve(state["question"], k=15)
    return {"chunks": chunks, "fallback_attempted": True}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _answer_is_insufficient(answer: str) -> bool:
    """Use Haiku to detect whether the answer signals insufficient context.

    More robust than string matching — catches any phrasing the model uses
    to express that it could not answer from the provided context.
    """
    response = _client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=5,
        system="Reply with only 'yes' or 'no'. No other text.",
        messages=[{
            "role": "user",
            "content": (
                "Does this answer indicate that the question could not be answered "
                "due to insufficient context or missing information?\n\n"
                f"Answer: {answer}"
            ),
        }],
    )
    return response.content[0].text.strip().lower() == "yes"


# ---------------------------------------------------------------------------
# Routing functions (conditional edges)
# ---------------------------------------------------------------------------

def route_after_grading(state: RAGState) -> str:
    """After grade_documents: rewrite if the grader rejected all chunks and not yet tried; else generate.

    Reads all_filtered (set by grade_documents before the safety floor) rather than
    checking state["chunks"], which is never empty after the safety floor runs.

    Returns:
        "rewrite_query"   — all_filtered is True AND rewrite hasn't been tried yet
        "generate_answer" — at least one chunk passed grading, OR rewrite already ran
                            (in which case generate_answer will return "I don't know")
    """
    if state.get("all_filtered", False) and not state.get("rewrite_attempted", False):
        return "rewrite_query"
    return "generate_answer"


def route_after_generate(state: RAGState) -> str:
    """After generate_answer: invoke the fallback path or stop.

    Returns:
        "fallback_retrieve" if the answer signals insufficient context
        and the fallback has not yet been attempted.
        END otherwise — including when the fallback already ran, so the
        graph terminates even if the fallback answer is also "I don't know".
    """
    already_tried = state.get("fallback_attempted", False)
    if not already_tried and _answer_is_insufficient(state["answer"]):
        return "fallback_retrieve"
    return END


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph():
    """Build and compile the RAG StateGraph."""
    builder = StateGraph(RAGState)

    builder.add_node("retrieve_rerank",   retrieve_rerank)
    builder.add_node("grade_documents",   grade_documents)
    builder.add_node("rewrite_query",     rewrite_query)
    builder.add_node("generate_answer",   generate_answer)
    builder.add_node("fallback_retrieve", fallback_retrieve)

    builder.add_edge(START, "retrieve_rerank")
    builder.add_edge("retrieve_rerank", "grade_documents")

    builder.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {"rewrite_query": "rewrite_query", "generate_answer": "generate_answer"},
    )

    builder.add_edge("rewrite_query", "retrieve_rerank")  # loop-back after rewrite

    builder.add_conditional_edges(
        "generate_answer",
        route_after_generate,
        {"fallback_retrieve": "fallback_retrieve", END: END},
    )
    builder.add_edge("fallback_retrieve", "generate_answer")

    return builder.compile()


# Module-level singleton — graph is built once at import, not on every ask() call
_graph = build_graph()
