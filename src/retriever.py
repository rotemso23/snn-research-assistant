"""
retriever.py — Load persisted ChromaDB and retrieve top-k chunks for a query.

Two retrieval strategies are available:
- retrieve(): MMR-based retrieval (diverse, reduces redundant chunks)
- retrieve_and_rerank(): MMR retrieval + CrossEncoder reranking (highest precision)
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

load_dotenv()

CHROMA_DIR = "chroma_db"
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_K = 15

_vectorstore: Chroma | None = None
_cross_encoder: CrossEncoder | None = None


def _get_vectorstore() -> Chroma:
    """Lazily load and cache the vectorstore."""
    global _vectorstore
    if _vectorstore is None:
        if not Path(CHROMA_DIR).exists():
            raise RuntimeError(
                f"ChromaDB not found at '{CHROMA_DIR}'. "
                "Run `python src/ingest.py --papers_dir papers/` first."
            )
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        _vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
        )
    return _vectorstore


def _get_cross_encoder() -> CrossEncoder:
    """Lazily load and cache the CrossEncoder reranking model."""
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    return _cross_encoder


def retrieve(query: str, k: int = DEFAULT_K) -> list[Document]:
    """Retrieve the top-k most relevant chunks using MMR for diversity."""
    vectorstore = _get_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": k * 3, "lambda_mult": 0.7},
    )
    return retriever.invoke(query)


def retrieve_and_rerank(query: str, fetch_k: int = 15, top_k: int = 5) -> list[Document]:
    """Retrieve fetch_k chunks with MMR, then rerank with CrossEncoder, return top_k.

    The CrossEncoder reads (query, chunk) pairs together for precise relevance
    scoring — more accurate than embedding cosine similarity alone.
    """
    docs = retrieve(query, k=fetch_k)
    ce = _get_cross_encoder()
    pairs = [[query, doc.page_content] for doc in docs]
    scores = ce.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_k]]


if __name__ == "__main__":
    test_query = "What is the main contribution of this thesis?"
    print(f"Query: {test_query}\n")
    results = retrieve(test_query)
    for i, doc in enumerate(results, 1):
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        print(f"--- Chunk {i} ({source}) ---")
        print(doc.page_content[:300])
        print()
