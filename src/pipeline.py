"""
pipeline.py — End-to-end RAG pipeline: retrieval + generation.
"""

from dotenv import load_dotenv
from src.retriever import retrieve_and_rerank
from src.generator import generate

load_dotenv()


def ask(question: str, k: int = 7, use_hyde: bool = True, multi_query: bool = True) -> dict:
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
    chunks = retrieve_and_rerank(question, fetch_k=20, top_k=k, use_hyde=use_hyde, multi_query=multi_query)
    return generate(question, chunks)


if __name__ == "__main__":
    question = "What are the main advantages of spiking neural networks over traditional ANNs?"
    print(f"Question: {question}\n")
    result = ask(question)
    print("Answer:", result["answer"])
    print("\nSources:", result["sources"])
