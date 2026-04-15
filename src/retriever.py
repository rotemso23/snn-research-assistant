"""
retriever.py — Load persisted ChromaDB and retrieve top-k chunks for a query.

Two retrieval strategies are available:
- retrieve(): MMR-based retrieval (diverse, reduces redundant chunks)
- retrieve_and_rerank(): MMR retrieval + CrossEncoder reranking (highest precision)
"""

import os
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

load_dotenv()

CHROMA_DIR = "chroma_db"
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
HYDE_MODEL = "claude-sonnet-4-6"
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



def _is_hebrew_dominant(text: str, threshold: float = 0.2) -> bool:
    """Return True if more than `threshold` fraction of letters are Hebrew."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    hebrew = sum(1 for c in letters if "\u05d0" <= c <= "\u05ea")
    return hebrew / len(letters) > threshold


def retrieve_from_source(query: str, source_substring: str, k: int = 10) -> list[Document]:
    """Retrieve top-k chunks from documents whose source path contains `source_substring`.

    ChromaDB metadata filters are unreliable through the LangChain wrapper, so we
    over-fetch and post-filter by source. Used to guarantee thesis representation
    when generic MMR is dominated by other papers. Hebrew chunks are also removed.
    """
    vectorstore = _get_vectorstore()
    try:
        # Use the collection count so we never request more than what exists.
        total = vectorstore._collection.count()
        fetch_n = min(total, max(k * 8, 50))
        results = vectorstore.similarity_search(query, k=fetch_n)
    except Exception:
        return []
    filtered = [
        doc for doc in results
        if source_substring.lower() in doc.metadata.get("source", "").lower()
        and not _is_hebrew_dominant(doc.page_content)
    ]
    return filtered[:k]


def retrieve(query: str, k: int = DEFAULT_K) -> list[Document]:
    """Retrieve the top-k most relevant chunks using MMR for diversity.

    Hebrew-dominant chunks (e.g. a Hebrew abstract in an otherwise-English
    thesis) are filtered out — the English embedding model embeds them poorly
    and they degrade both retrieval ranking and generated answers.
    """
    vectorstore = _get_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": k * 3, "lambda_mult": 0.7},
    )
    docs = retriever.invoke(query)
    return [doc for doc in docs if not _is_hebrew_dominant(doc.page_content)]


def _generate_query_variants(question: str, n: int = 2) -> list[str]:
    """Generate n alternative phrasings of the question for multi-query retrieval.

    Different phrasings embed differently and surface different chunks, so the
    merged candidate pool is much richer than any single query alone.
    """
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=HYDE_MODEL,
        max_tokens=150,
        system=(
            "You are a search query optimizer for academic paper retrieval. "
            "Given a question, write alternative phrasings that use different vocabulary "
            "and angle — the kind of language that would appear in a research paper or thesis. "
            f"Output exactly {n} alternatives, one per line, no numbering or extra text."
        ),
        messages=[{"role": "user", "content": question}],
    )
    lines = [l.strip() for l in response.content[0].text.strip().split("\n") if l.strip()]
    return lines[:n]


def _generate_hypothetical_answer(question: str) -> str:
    """Generate a plausible hypothetical answer to use as the retrieval query (HyDE).

    The hypothetical answer embeds closer to real paper chunks than the bare
    question does, because it uses the same dense, declarative language as the
    source documents. The answer is never shown to the user — it is only used
    to compute the search embedding.
    """
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=HYDE_MODEL,
        max_tokens=150,
        system=(
            "You are a research assistant specializing in spiking neural networks and "
            "neuromorphic computing. Given a question, write a concise 2-3 sentence "
            "answer as it might appear in a research paper — dense, technical, and "
            "declarative. Output only the answer, nothing else."
        ),
        messages=[{"role": "user", "content": question}],
    )
    return response.content[0].text.strip()



_THESIS_KEYWORDS = {"thesis", "your work", "your paper", "this work", "rotem", "solomon"}


def retrieve_and_rerank(
    query: str,
    fetch_k: int = 15,
    top_k: int = 5,
    use_hyde: bool = False,
    multi_query: bool = False,
) -> list[Document]:
    """Retrieve fetch_k chunks with MMR, then rerank with CrossEncoder, return top_k.

    The CrossEncoder reads (query, chunk) pairs together for precise relevance
    scoring — more accurate than embedding cosine similarity alone.

    Args:
        query:        The user's original question.
        fetch_k:      Number of candidates to fetch per query via MMR.
        top_k:        Number of chunks to return after reranking.
        use_hyde:     If True, generate a hypothetical answer and use its embedding
                      for the MMR search instead of the raw question embedding.
                      The CrossEncoder always uses the original query.
        multi_query:  If True, generate 2 alternative phrasings and retrieve
                      candidates for all 3 queries. The merged pool (deduplicated)
                      is reranked together — widens coverage for hard retrieval cases.
    """
    if multi_query:
        queries = [query] + _generate_query_variants(query, n=2)
        seen: dict[str, Document] = {}
        for q in queries:
            retrieval_q = _generate_hypothetical_answer(q) if use_hyde else q
            for doc in retrieve(retrieval_q, k=fetch_k):
                key = doc.page_content[:200]
                if key not in seen:
                    seen[key] = doc
        candidates = list(seen.values())
    else:
        retrieval_query = _generate_hypothetical_answer(query) if use_hyde else query
        candidates = retrieve(retrieval_query, k=fetch_k)

    # When the query is clearly about the thesis itself, guarantee thesis chunks
    # are in the candidate pool — generic MMR often fills the pool with other papers.
    query_lower = query.lower()
    thesis_guaranteed: list[Document] = []
    if any(kw in query_lower for kw in _THESIS_KEYWORDS):
        seen_keys = {doc.page_content[:200] for doc in candidates}
        thesis_boost = retrieve_from_source(query, "thesis", k=10)
        for doc in thesis_boost:
            key = doc.page_content[:200]
            if key not in seen_keys:
                candidates.append(doc)
                seen_keys.add(key)
        # Pin the top-3 thesis chunks into the final result so the CrossEncoder
        # cannot displace them entirely in favour of other papers.
        thesis_guaranteed = thesis_boost[:3]

    ce = _get_cross_encoder()
    # Always rerank against the original question, not the hypothetical.
    pairs = [[query, doc.page_content] for doc in candidates]
    scores = ce.predict(pairs)
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    top = [doc for _, doc in ranked[:top_k]]

    # Merge guaranteed thesis chunks: add any that didn't make the cut, replacing
    # the lowest-ranked slots so total count stays at top_k.
    if thesis_guaranteed:
        top_keys = {doc.page_content[:200] for doc in top}
        for doc in thesis_guaranteed:
            if doc.page_content[:200] not in top_keys and len(top) >= top_k:
                top.pop()  # drop lowest-ranked (last after sort)
            if doc.page_content[:200] not in top_keys:
                top.append(doc)
                top_keys.add(doc.page_content[:200])

    return top


if __name__ == "__main__":
    test_query = "What is the main contribution of this thesis?"
    print(f"Query: {test_query}\n")
    results = retrieve(test_query)
    for i, doc in enumerate(results, 1):
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        print(f"--- Chunk {i} ({source}) ---")
        print(doc.page_content[:300])
        print()
