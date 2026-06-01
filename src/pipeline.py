"""
pipeline.py — Public API for the SNN RAG pipeline.

Delegates to the agentic LangGraph StateGraph defined in src/agent_graph.py.
The ask() signature and return type are identical to the previous version —
app.py, evaluate.py, and the HuggingFace deployment are untouched.

src/graph.py (the original CRAG pipeline) is kept intact so that
evaluate.py --grading can still invoke it directly.
"""

from dotenv import load_dotenv
from src.agent_graph import _agent_graph

load_dotenv()


def ask(question: str, k: int = 10, use_hyde: bool = True, multi_query: bool = True) -> dict:
    """
    Ask a question over the ingested papers.

    Args:
        question:     The user's question.
        k:            Number of top chunks to pass to the generator after reranking.
        use_hyde:     If True, use HyDE (Hypothetical Document Embeddings) for
                      retrieval — generates a plausible answer first and searches
                      with its embedding, improving context recall.
        multi_query:  If True, generate 2 alternative phrasings and merge retrieval
                      candidates before reranking — improves coverage for hard queries.

    Returns:
        {"answer": str, "sources": list[str]}
    """
    initial_state = {
        # ── v5 fields (unchanged) ─────────────────────────────────────────
        "question":        question,
        "k":               k,
        "use_hyde":        use_hyde,
        "multi_query":     multi_query,
        # retrieval_query starts equal to question; grade_node may overwrite with missing_aspects
        "retrieval_query": question,
        # Remaining fields initialised here because LangGraph TypedDict
        # channels have no runtime defaults.
        "chunks":          [],
        "answer":          "",
        "sources":         [],
        "need_more":       False,
        "retry_count":     0,
        # ── v7 new fields ─────────────────────────────────────────────────
        "query_type":  "",
        "sub_queries": [],
    }
    final_state = _agent_graph.invoke(initial_state)
    return {"answer": final_state["answer"], "sources": final_state["sources"]}


if __name__ == "__main__":
    question = "What are the main advantages of spiking neural networks over traditional ANNs?"
    print(f"Question: {question}\n")
    result = ask(question)
    print("Answer:", result["answer"])
    print("\nSources:", result["sources"])
