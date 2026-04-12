"""
pipeline.py — End-to-end RAG pipeline: retrieval + generation.
"""

from dotenv import load_dotenv
from src.retriever import retrieve_and_rerank
from src.generator import generate

load_dotenv()


def ask(question: str, k: int = 7) -> dict:
    """
    Ask a question over the ingested papers.

    Args:
        question: The user's question.
        k: Number of top chunks to pass to the generator after reranking.

    Returns:
        {"answer": str, "sources": list[str]}
    """
    chunks = retrieve_and_rerank(question, fetch_k=20, top_k=k)
    return generate(question, chunks)


if __name__ == "__main__":
    question = "What are the main advantages of spiking neural networks over traditional ANNs?"
    print(f"Question: {question}\n")
    result = ask(question)
    print("Answer:", result["answer"])
    print("\nSources:", result["sources"])
