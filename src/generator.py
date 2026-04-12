"""
generator.py — Send retrieved context to Claude and return an answer with citations.
"""

import os
from dotenv import load_dotenv
import anthropic
from langchain_core.documents import Document

load_dotenv()

CLAUDE_MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are a research assistant specializing in Spiking Neural Networks (SNNs), \
neuromorphic computing, and low-power deep learning.

Answer the user's question using ONLY the context passages provided below. Follow these rules:
1. Base your answer solely on the provided context — do not use outside knowledge.
2. For each claim, cite the source document by its filename in parentheses, e.g. (thesis_rotem_solomon.pdf).
3. If the context does not contain enough information to answer the question, respond with:
   "I don't know — the provided context does not contain enough information to answer this question."
4. Be concise and precise."""


def _build_context_block(chunks: list[Document]) -> str:
    """Format retrieved chunks into a numbered context block."""
    parts = []
    for i, doc in enumerate(chunks, 1):
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        parts.append(f"[{i}] Source: {source}\n{doc.page_content}")
    return "\n\n".join(parts)


def _extract_sources(chunks: list[Document]) -> list[str]:
    """Return deduplicated list of source filenames."""
    seen = set()
    sources = []
    for doc in chunks:
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        if source not in seen:
            seen.add(source)
            sources.append(source)
    return sources


def generate(question: str, chunks: list[Document]) -> dict:
    """
    Call Claude with retrieved context and return answer + sources.

    Returns:
        {"answer": str, "sources": list[str]}
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set. Check your .env file.")

    client = anthropic.Anthropic(api_key=api_key)

    context_block = _build_context_block(chunks)
    user_message = f"Context:\n{context_block}\n\nQuestion: {question}"

    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    answer = message.content[0].text
    sources = _extract_sources(chunks)

    return {"answer": answer, "sources": sources}


if __name__ == "__main__":
    from src.retriever import retrieve

    question = "What spiking neuron model does the thesis use?"
    print(f"Question: {question}\n")
    chunks = retrieve(question)
    result = generate(question, chunks)
    print("Answer:", result["answer"])
    print("\nSources:", result["sources"])
