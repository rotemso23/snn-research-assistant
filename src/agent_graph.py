"""
agent_graph.py — Agentic RAG pipeline using LangGraph (v7 architecture).

Graph topology (5 nodes, 1 visible back-edge):

    START
      │
      ▼
    [router_node]  ── route_after_router ─── "multi_part" ──► [decompose_node]
      │                                                               │
      │ (thesis_specific / conceptual_snn / simple_factual)          │
      └───────────────────────────────────────────────────────────────┘
                                                                       │
                                                                       ▼
                          ┌──────────────── [retrieve_rerank] ◄────────┘ ◄──┐
                          │                        │                         │
                          │                        ▼                         │ need_more=True
                          │                  [grade_node]                    │ retry_count < 2
                          │                        │                         │
                          │              route_after_grade ──────────────────┘
                          │                        │
                          │               need_more=False
                          │                        ▼
                          │                  [answer_node]
                          │                        │
                          │                        ▼
                          │                       END
                          │
                          └─ (decompose_node flows here)

v7 vs v6:
  - Removed faithfulness_check_node — Haiku/RAGAS score misalignment caused
    strict-prompt retries to hurt faithfulness rather than help it.
  - Router: thesis_specific no longer disables HyDE — data showed HyDE improves
    recall for thesis questions by embedding closer to paper vocabulary.
  - Only simple_factual disables HyDE and multi_query.

v7 vs v5:
  router_node     NEW — Haiku classifies query intent; simple_factual disables HyDE/multi_query
  decompose_node  NEW — Haiku decomposes multi_part questions into sub_queries
  retrieve_rerank UPDATED — handles sub_queries: fan-out + CrossEncoder merge
  grade_node      UPDATED — needs_more_context (bool) replaces sufficiency_score threshold
  MAX_RETRIES     1 → 2
"""

from typing_extensions import TypedDict
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END
import anthropic

from src.retriever import retrieve_and_rerank, _get_cross_encoder  # _get_cross_encoder used in sub-query merge
from src.generator import (
    _build_context_block,
    SYSTEM_PROMPT,
    generate,
)

# ── Singleton client ──────────────────────────────────────────────────────────
_client = anthropic.Anthropic()

# ── Constants ─────────────────────────────────────────────────────────────────
HAIKU_MODEL = "claude-haiku-4-5-20251001"   # cheap grader/router
MAX_RETRIES = 2    # circuit breaker — caps retrieval attempts when the corpus lacks the answer


# ── State ─────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    # ── Inputs — set at graph entry ──────────────────────────────────────────
    question:        str
    k:               int
    use_hyde:        bool
    multi_query:     bool
    # ── Mutable retrieval state ──────────────────────────────────────────────
    retrieval_query: str           # starts = question; grade_node may overwrite with missing_aspects
    chunks:          list[Document]
    # ── Answer output ────────────────────────────────────────────────────────
    answer:          str
    sources:         list[str]
    # ── Retrieval-retry circuit-breaker ──────────────────────────────────────
    need_more:       bool
    retry_count:     int           # incremented by grade_node on each retry
    # ── Router/decompose ─────────────────────────────────────────────────────
    query_type:      str            # set by router_node; default ""
    sub_queries:     list[str]     # set by decompose_node; default []


# ── Tool definitions ──────────────────────────────────────────────────────────

ROUTER_TOOL = {
    "name": "classify_query",
    "description": "Classify the user's question to determine the optimal retrieval strategy.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query_type": {
                "type": "string",
                "enum": ["simple_factual", "thesis_specific", "conceptual_snn", "multi_part"],
                "description": (
                    "simple_factual: short question with one clear factual answer "
                    "(e.g. 'What neuron model is used?'). "
                    "thesis_specific: asks about this specific thesis — its goals, models, datasets, "
                    "contributions, authors, or results "
                    "(e.g. 'What is the main goal of the thesis?', 'What models were proposed?'). "
                    "conceptual_snn: asks about SNN theory or general neuromorphic concepts — "
                    "not specific to this thesis "
                    "(e.g. 'Why are SNNs energy efficient?', 'What is the Boltzmann distribution?'). "
                    "multi_part: compound question with two or more distinct sub-questions that "
                    "require separate retrieval "
                    "(e.g. 'What models were proposed and how were they evaluated?')."
                ),
            },
        },
        "required": ["query_type"],
    },
}

DECOMPOSE_TOOL = {
    "name": "decompose_query",
    "description": (
        "Break a multi-part question into focused sub-queries, each targeting "
        "a distinct piece of information independently retrievable from an academic paper."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "sub_queries": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
                "maxItems": 3,
                "description": (
                    "2–3 independent search queries derived from the original question. "
                    "Each sub-query must be self-contained and retrievable on its own. "
                    "Use academic/technical vocabulary. Do not overlap in intent."
                ),
            },
        },
        "required": ["sub_queries"],
    },
}

GRADE_TOOL = {
    "name": "grade_context",
    "description": (
        "Grade the relevance of retrieved chunks and decide whether more retrieval is needed."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "relevant_chunk_indices": {
                "type": "array",
                "items": {"type": "integer"},
                "description": (
                    "1-based indices of chunks that help answer the question, even partially. "
                    "Only exclude a chunk if it is completely off-topic. When in doubt, include it."
                ),
            },
            "needs_more_context": {
                "type": "boolean",
                "description": (
                    "True if the selected chunks are insufficient to answer the question "
                    "and a fresh retrieval attempt should be made. "
                    "False if the selected chunks provide enough for a useful answer, "
                    "even if the coverage is partial."
                ),
            },
            "missing_aspects": {
                "type": "string",
                "description": (
                    "Only provide when needs_more_context=True. "
                    "Describe the specific information that is missing using technical vocabulary "
                    "(model names, algorithm names, dataset names, physics terms) that would appear "
                    "in a research paper. This text will be used directly as a retrieval query, "
                    "so be specific and information-dense, not conversational."
                ),
            },
        },
        "required": ["relevant_chunk_indices", "needs_more_context"],
    },
}


# ── System prompts ────────────────────────────────────────────────────────────

ROUTER_SYSTEM_PROMPT = (
    "You are a query classifier for a RAG pipeline on spiking neural networks (SNNs) "
    "and neuromorphic computing.\n\n"
    "Classify the user's question into exactly one category using the classify_query tool:\n"
    "- thesis_specific: questions about this specific thesis/work — its goals, proposed models, "
    "datasets used, results, authors, or contributions.\n"
    "- conceptual_snn: questions about SNN theory or general neuromorphic concepts not tied to "
    "a specific thesis (membrane potential, Boltzmann distribution, energy efficiency in general).\n"
    "- simple_factual: very short questions with a single direct factual answer.\n"
    "- multi_part: compound questions containing two or more distinct sub-questions that each "
    "require separate retrieval (look for 'and', 'as well as', multiple question marks)."
)

DECOMPOSE_SYSTEM_PROMPT = (
    "You are a query decomposer for a RAG system on spiking neural networks and neuromorphic computing.\n\n"
    "Break the compound question into 2–3 focused sub-questions, each independently retrievable "
    "from an academic paper. Use technical and academic vocabulary. "
    "Each sub-query must be self-contained — do not refer to 'the above' or 'the thesis' vaguely. "
    "Use the decompose_query tool."
)

GRADE_SYSTEM_PROMPT = (
    "You are a relevance judge for a RAG pipeline on spiking neural networks (SNNs) "
    "and neuromorphic computing.\n\n"
    "Step 1 — Filter chunks: include each numbered chunk if it helps answer ANY part of "
    "the question, even partially. Only exclude chunks that are completely off-topic. "
    "When in doubt, include.\n\n"
    "Step 2 — Decide: set needs_more_context=True ONLY if the selected chunks are clearly "
    "insufficient to give a useful answer — for example, they cover none of the key aspects "
    "of the question. Set needs_more_context=False if the chunks provide enough for a "
    "reasonable answer, even if coverage is partial.\n\n"
    "If needs_more_context=True, also provide missing_aspects: a dense, technical description "
    "of what is missing (model names, algorithm names, dataset names, physics terms — "
    "this text will be used directly as a retrieval query).\n\n"
    "Call grade_context with the relevant indices, your decision, and optionally missing_aspects."
)

# Extends generator.py's SYSTEM_PROMPT — rule 4 tightened: concise + no unnecessary hedging
AGENT_ANSWER_SYSTEM_PROMPT = SYSTEM_PROMPT.rstrip().replace(
    "4. Be concise and precise.",
    "4. Be concise and direct. Do not hedge unnecessarily if the context contains the answer.",
)


# ── Node functions ────────────────────────────────────────────────────────────

def router_node(state: AgentState) -> dict:
    """Node 1: Classify query intent and override retrieval strategy flags.

    Haiku classifies the question into one of four types:
      simple_factual   → use_hyde=False, multi_query=False (short direct questions)
      thesis_specific  → keep caller's use_hyde / multi_query; force_thesis_boost in retrieval
      conceptual_snn   → keep caller's use_hyde / multi_query
      multi_part       → keep caller's use_hyde / multi_query; route_after_router → decompose_node

    Writing use_hyde and multi_query back into state allows downstream nodes to read
    the router-adjusted strategy without any additional plumbing.
    """
    response = _client.messages.create(
        model=HAIKU_MODEL,
        max_tokens=64,
        system=[
            {
                "type": "text",
                "text": ROUTER_SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        tools=[ROUTER_TOOL],
        tool_choice={"type": "tool", "name": "classify_query"},
        messages=[{"role": "user", "content": state["question"]}],
    )

    for block in response.content:
        if block.type == "tool_use" and block.name == "classify_query":
            query_type = block.input.get("query_type", "conceptual_snn")

            if query_type == "simple_factual":
                # Short direct question — skip expensive query expansion
                use_hyde    = False
                multi_query = False
            elif query_type in ("thesis_specific", "conceptual_snn", "multi_part"):
                # Keep caller's flags — HyDE helps recall for all non-trivial question types
                use_hyde    = state["use_hyde"]
                multi_query = state["multi_query"]
            else:
                # Unexpected value — safe default
                query_type  = "conceptual_snn"
                use_hyde    = state["use_hyde"]
                multi_query = state["multi_query"]

            return {
                "query_type":  query_type,
                "use_hyde":    use_hyde,
                "multi_query": multi_query,
            }

    # Unexpected response shape — proceed with original flags
    return {"query_type": "conceptual_snn"}


def decompose_node(state: AgentState) -> dict:
    """Node 2 (conditional): Decompose a multi_part question into sub_queries.

    Only reached when route_after_router returns 'decompose_node'.
    Sets state['sub_queries'] which retrieve_rerank_node uses for fan-out retrieval.
    Falls back to [] on malformed tool response, which causes retrieve_rerank_node
    to fall through to the standard single-query path.
    """
    response = _client.messages.create(
        model=HAIKU_MODEL,
        max_tokens=200,
        system=[
            {
                "type": "text",
                "text": DECOMPOSE_SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        tools=[DECOMPOSE_TOOL],
        tool_choice={"type": "tool", "name": "decompose_query"},
        messages=[{"role": "user", "content": state["question"]}],
    )

    for block in response.content:
        if block.type == "tool_use" and block.name == "decompose_query":
            sub_queries = block.input.get("sub_queries", [])
            if 2 <= len(sub_queries) <= 3:
                return {"sub_queries": sub_queries}
            # Malformed (wrong count) — fall through to standard path
            return {"sub_queries": []}

    return {"sub_queries": []}


def retrieve_rerank_node(state: AgentState) -> dict:
    """Node 3: HyDE + multi-query (or sub-query fan-out) + CrossEncoder reranking.

    Two retrieval paths:
    1. Sub-query path (sub_queries non-empty AND retry_count == 0):
       - Calls retrieve_and_rerank() per sub-query (multi_query=False each)
       - Merges unique chunks (by first 200 chars)
       - Final CrossEncoder re-ranks merged pool against original question
    2. Standard path (all other cases, including grade retries):
       - Delegates to retrieve_and_rerank() unchanged from v5
       - Uses state['retrieval_query'] (may be missing_aspects from grade_node)
       - HyDE skipped after first pass (same as v5)

    The retry_count==0 guard on the sub-query branch ensures grade retries always
    use the standard path with the grader's missing_aspects query.
    """
    sub_queries  = state.get("sub_queries", [])
    retry_count  = state["retry_count"]
    use_hyde_now = state["use_hyde"] and retry_count <= 1
    force_boost  = state.get("query_type") == "thesis_specific"

    if sub_queries and retry_count == 0:
        # Fan-out: retrieve per sub-query, merge, re-rank merged pool
        seen: dict[str, Document] = {}
        for sq in sub_queries:
            for doc in retrieve_and_rerank(
                sq,
                fetch_k=35,
                top_k=state["k"],
                use_hyde=use_hyde_now,
                multi_query=False,  # already decomposed — no further expansion
                force_thesis_boost=force_boost,
            ):
                key = doc.page_content[:200]
                if key not in seen:
                    seen[key] = doc

        merged = list(seen.values())
        if not merged:
            # Fallback: nothing retrieved — try original question via standard path
            return {"chunks": retrieve_and_rerank(
                state["retrieval_query"], fetch_k=35, top_k=state["k"],
                use_hyde=use_hyde_now, multi_query=state["multi_query"],
                force_thesis_boost=force_boost,
            )}

        # Re-rank merged pool against the ORIGINAL question (not sub-queries)
        ce = _get_cross_encoder()  # noqa: private import — documented dependency
        pairs  = [[state["question"], doc.page_content] for doc in merged]
        scores = ce.predict(pairs)
        ranked = sorted(zip(scores, merged), key=lambda x: x[0], reverse=True)
        return {"chunks": [doc for _, doc in ranked[:state["k"]]]}

    # Standard path (v5 behaviour)
    chunks = retrieve_and_rerank(
        state["retrieval_query"],
        fetch_k=35,
        top_k=state["k"],
        use_hyde=use_hyde_now,
        multi_query=state["multi_query"],
        force_thesis_boost=force_boost,
    )
    return {"chunks": chunks}


def grade_node(state: AgentState) -> dict:
    """Node 4: Haiku grades chunks and decides directly whether more retrieval is needed.

    Calls grade_context tool (tool_choice=required) to obtain:
      - relevant_chunk_indices: which chunks help answer the question
      - needs_more_context (bool): agent's direct yes/no decision — no external threshold
      - missing_aspects (optional): what's missing — used as next retrieval_query

    Retry if needs_more_context=True AND retry budget remains (MAX_RETRIES is a
    circuit breaker, not the primary stopping condition — the agent decides).
    missing_aspects is truncated to 400 chars to carry sufficient retrieval signal.

    Full-reject handling: if Haiku marks zero chunks relevant, forces a retry rather
    than silently passing garbage context to the generator.
    """
    retry  = state["retry_count"]
    chunks = state["chunks"]

    if not chunks:
        return {"need_more": False}

    retry_hint = (
        f"\n\n[This is retrieval attempt {retry + 1}/{MAX_RETRIES + 1}. "
        "Only set needs_more_context=True if something critical is genuinely missing. "
        "If the chunks cover the main points of the question, set needs_more_context=False.]"
        if retry > 0 else ""
    )

    response = _client.messages.create(
        model=HAIKU_MODEL,
        max_tokens=512,
        system=[
            {
                "type": "text",
                "text": GRADE_SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        tools=[GRADE_TOOL],
        tool_choice={"type": "tool", "name": "grade_context"},
        messages=[
            {
                "role": "user",
                "content": (
                    f"{_build_context_block(chunks)}\n\n"
                    f"Question: {state['question']}"
                    f"{retry_hint}"
                ),
            }
        ],
    )

    for block in response.content:
        if block.type == "tool_use" and block.name == "grade_context":
            indices         = block.input.get("relevant_chunk_indices", [])
            needs_more      = block.input.get("needs_more_context", False)
            missing_aspects = block.input.get("missing_aspects")

            graded = [
                chunks[i - 1]
                for i in indices
                if isinstance(i, int) and 1 <= i <= len(chunks)
            ]
            if not graded:
                # Full-reject: Haiku said nothing is useful — force a fresh retry.
                if retry < MAX_RETRIES:
                    return {
                        "chunks":          chunks,            # kept only as fallback state
                        "need_more":       True,
                        "retry_count":     retry + 1,
                        "retrieval_query": state["question"], # reset to original — missing_aspects failed
                    }
                # Retry budget exhausted — last resort: pass everything
                graded = chunks

            insufficient = needs_more and retry < MAX_RETRIES
            updates: dict = {
                "chunks":      graded,
                "need_more":   insufficient,
                "retry_count": retry + 1 if insufficient else retry,
            }
            if insufficient and missing_aspects:
                # Truncate to 400 chars — carries more retrieval signal than the old 200-char limit
                updates["retrieval_query"] = missing_aspects[:400]
            return updates

    # Unexpected response shape — proceed to answer with whatever chunks we have
    return {"need_more": False}


def answer_node(state: AgentState) -> dict:
    """Node 5: Generate the final answer — no tools, pure system-prompt completion."""
    result = generate(
        state["question"],
        state["chunks"],
        system_prompt=AGENT_ANSWER_SYSTEM_PROMPT,
    )
    return {"answer": result["answer"], "sources": result["sources"]}


# ── Routing functions ─────────────────────────────────────────────────────────

def route_after_router(state: AgentState) -> str:
    """Route to decompose_node for multi_part queries; all others go to retrieval."""
    if state.get("query_type") == "multi_part":
        return "decompose_node"
    return "retrieve_rerank"


def route_after_grade(state: AgentState) -> str:
    """Route back to retrieval if grade_node flagged insufficient context."""
    if state.get("need_more") and state.get("retry_count", 0) <= MAX_RETRIES:
        return "retrieve_rerank"
    return "answer_node"


# ── Graph construction ────────────────────────────────────────────────────────

def build_agent_graph():
    """Build and compile the v7 agentic RAG StateGraph."""
    builder = StateGraph(AgentState)

    builder.add_node("router_node",    router_node)
    builder.add_node("decompose_node", decompose_node)
    builder.add_node("retrieve_rerank", retrieve_rerank_node)
    builder.add_node("grade_node",     grade_node)
    builder.add_node("answer_node",    answer_node)

    # START → router
    builder.add_edge(START, "router_node")

    # router → decompose (multi_part) or retrieve_rerank (everything else)
    builder.add_conditional_edges(
        "router_node",
        route_after_router,
        {"decompose_node": "decompose_node", "retrieve_rerank": "retrieve_rerank"},
    )

    # decompose → retrieve_rerank (always)
    builder.add_edge("decompose_node", "retrieve_rerank")

    # retrieve_rerank → grade (always)
    builder.add_edge("retrieve_rerank", "grade_node")

    # grade → retrieval retry OR answer
    builder.add_conditional_edges(
        "grade_node",
        route_after_grade,
        {"retrieve_rerank": "retrieve_rerank", "answer_node": "answer_node"},
    )

    # answer → END
    builder.add_edge("answer_node", END)

    return builder.compile()


# Module-level singleton — built once at import time
_agent_graph = build_agent_graph()
