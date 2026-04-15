"""
diagnose_retrieval.py — Print retrieved chunks for the 3 failing eval questions.

Usage:
    venv/Scripts/python diagnose_retrieval.py
"""

import io
import os
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from dotenv import load_dotenv
from src.retriever import retrieve_and_rerank

load_dotenv()

QUESTIONS = [
    "What spiking neuron model is used in the thesis?",
    "What models were proposed in the thesis?",
    "What biological principles inspire the proposed models?",
]

GROUND_TRUTHS = [
    "The thesis uses the Leaky Integrate-and-Fire (LIF) neuron model. In this model, the membrane potential accumulates input current over time and emits a spike when it crosses a threshold, after which it resets.",
    "The thesis proposes four new AI models: the Linear Model, the Exponential Model, the Scaled Exponential Model, and the Split Scaled Exponential Model. These models are designed to be both energy-efficient and suitable for neuromorphic hardware implementation.",
    "The proposed models draw inspiration from two fundamental properties of brain computation: spike-based communication between neurons, and exponential or logarithmic domain computation arising from ion energy following a Boltzmann distribution.",
]

SEP = "-" * 70

# ── Boltzmann keyword scan ───────────────────────────────────────────────────
print("\n" + "=" * 70)
print("KEYWORD SCAN: searching all ChromaDB chunks for 'Boltzmann'")
print(SEP)

from src.retriever import _get_vectorstore
vs = _get_vectorstore()
collection = vs._collection
all_docs = collection.get(include=["documents", "metadatas"])

boltzmann_hits = [
    (all_docs["metadatas"][i], all_docs["documents"][i])
    for i, doc in enumerate(all_docs["documents"])
    if "boltzmann" in doc.lower()
]

if boltzmann_hits:
    print(f"Found {len(boltzmann_hits)} chunk(s) containing 'Boltzmann':\n")
    for meta, content in boltzmann_hits:
        source = os.path.basename(meta.get("source", "unknown"))
        page = meta.get("page", "?")
        print(f"[{source} — page {page}]")
        print(content[:600])
        if len(content) > 600:
            print("... [truncated]")
        print(SEP)
else:
    print("No chunks contain the word 'Boltzmann' — content may not be ingested.")

# ── Targeted embedding search ────────────────────────────────────────────────
print("\n" + "=" * 70)
print("TARGETED SEARCH: 'Boltzmann distribution ion energy neuron biological'")
print(SEP)
targeted_chunks = retrieve_and_rerank(
    "Boltzmann distribution ion energy neuron biological inspiration",
    fetch_k=20, top_k=5, use_hyde=False, multi_query=False,
)
for i, doc in enumerate(targeted_chunks, 1):
    source = os.path.basename(doc.metadata.get("source", "unknown"))
    page = doc.metadata.get("page", "?")
    print(f"\n[Chunk {i}] {source} — page {page}")
    print(doc.page_content[:500])
    print(SEP)

# ── Original 3-question diagnostic ──────────────────────────────────────────
for question, ground_truth in zip(QUESTIONS, GROUND_TRUTHS):
    print(f"\n{'='*70}")
    print(f"QUESTION: {question}")
    print(f"\nGROUND TRUTH:\n{ground_truth}")
    print(f"\nRETRIEVED CHUNKS (top 7, with HyDE):")
    print(SEP)

    chunks = retrieve_and_rerank(question, fetch_k=20, top_k=7, use_hyde=True, multi_query=True)
    for i, doc in enumerate(chunks, 1):
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        page = doc.metadata.get("page", "?")
        print(f"\n[Chunk {i}] {source} — page {page}")
        print(doc.page_content[:600])
        if len(doc.page_content) > 600:
            print("... [truncated]")
        print(SEP)
