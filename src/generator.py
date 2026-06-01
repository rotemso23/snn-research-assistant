"""
generator.py — Send retrieved context to Claude and return an answer with citations.
"""

import os
import re
from dotenv import load_dotenv
import anthropic
from langchain_core.documents import Document

load_dotenv()

_client = anthropic.Anthropic()

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


def _extract_cited_sources(answer: str, chunks: list[Document]) -> list[str]:
    """Return sources the model actually cited in its answer, in order of appearance.

    Parses (filename.pdf) patterns from the answer text and cross-references
    against the chunk sources to guard against hallucinated filenames.
    Falls back to all chunk sources if the model cited nothing (e.g. ignored
    the citation instruction).
    """
    valid_sources = {
        os.path.basename(doc.metadata.get("source", "unknown"))
        for doc in chunks
    }
    cited = re.findall(r'\(([^)]+\.pdf)\)', answer)
    seen: set[str] = set()
    sources: list[str] = []
    for s in cited:
        if s in valid_sources and s not in seen:
            seen.add(s)
            sources.append(s)
    # Fallback: model answered without citing — return all chunk sources
    return sources if sources else list(valid_sources)


def generate(question: str, chunks: list[Document], system_prompt: str | None = None) -> dict:
    """
    Call Claude with retrieved context and return answer + sources.

    Args:
        question:      The user's question.
        chunks:        Retrieved context documents.
        system_prompt: Optional override for SYSTEM_PROMPT. Defaults to the
                       module-level SYSTEM_PROMPT when not provided.

    Returns:
        {"answer": str, "sources": list[str]}
    """
    if not chunks:
        return {
            "answer": "I don't know — the provided context does not contain enough information to answer this question.",
            "sources": [],
        }

    prompt = system_prompt if system_prompt is not None else SYSTEM_PROMPT
    context_block = _build_context_block(chunks)
    user_message = f"Context:\n{context_block}\n\nQuestion: {question}"

    message = _client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system=[
            {
                "type": "text",
                "text": prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_message}],
    )

    answer = message.content[0].text
    sources = _extract_cited_sources(answer, chunks)

    return {"answer": answer, "sources": sources}


if __name__ == "__main__":
    from src.retriever import retrieve

    question = "What spiking neuron model does the thesis use?"
    print(f"Question: {question}\n")
    chunks = retrieve(question)
    result = generate(question, chunks)
    print("Answer:", result["answer"])
    print("\nSources:", result["sources"])
