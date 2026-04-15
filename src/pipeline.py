"""
pipeline.py — End-to-end RAG pipeline: retrieval + generation.
"""

from dotenv import load_dotenv
from src.retriever import retrieve, retrieve_and_rerank
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
    result = generate(question, chunks)

    # Fallback: HyDE + CrossEncoder misses structural chunks (ToC entries, section
    # headers) because natural-language questions don't embed close to navigation
    # content. Augmenting with domain terms ("spiking neural network thesis") shifts
    # the embedding toward the thesis's own structural chunks.
    if "does not contain enough information" in result["answer"]:
        augmented = question + " spiking neural network thesis"
        chunks = retrieve(augmented, k=15)
        result = generate(question, chunks)

    return result


if __name__ == "__main__":
    question = "What are the main advantages of spiking neural networks over traditional ANNs?"
    print(f"Question: {question}\n")
    result = ask(question)
    print("Answer:", result["answer"])
    print("\nSources:", result["sources"])
