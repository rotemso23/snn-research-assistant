---
title: SNN Research Assistant
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: streamlit
app_file: app.py
pinned: false
---

# SNN Research Assistant

A **RAG (Retrieval-Augmented Generation) pipeline** for exploring academic papers on Spiking Neural Networks, neuromorphic computing, and low-power deep learning — the domain of my M.Sc. thesis at the Technion.

Ask natural-language questions, get answers grounded in the papers with source citations. The system says "I don't know" when the context doesn't support an answer.

**[Live Demo on HuggingFace Spaces](https://huggingface.co/spaces/rotemso23/snn-research-assistant)**

---

## Architecture

```
User question
      │
      ▼
┌─────────────────────┐
│  Multi-Query        │   Claude generates 2 alternative phrasings
│  (query expansion)  │   to widen the candidate pool
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  HyDE               │   Claude generates a hypothetical answer
│  (query expansion)  │   per query to improve embedding alignment
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  HuggingFace        │   BAAI/bge-large-en-v1.5
│  Embeddings         │   (hypothetical answer embeddings)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  ChromaDB           │   MMR retrieval × 3 queries
│  Vector Store       │   fetch_k=20 per query, results merged
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  CrossEncoder       │   ms-marco-MiniLM-L-6-v2
│  Reranker           │   top_k=7 chunks selected from merged pool
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Claude API         │   claude-sonnet-4-6
│  Generation         │   answers from context only
└────────┬────────────┘
         │
         ▼
  "I don't know"? ──► Fallback: augment query with domain terms,
         │                       retry with plain MMR (no HyDE, no CrossEncoder)
         │                       surfaces ToC/structural chunks reliably
         ▼
  Answer + Sources
```

---

## Tech Stack

| Layer | Tool | Why |
|-------|------|-----|
| Orchestration | LangChain | Industry-standard RAG framework |
| Embeddings | `BAAI/bge-large-en-v1.5` (HuggingFace) | State-of-the-art retrieval embeddings |
| Vector store | ChromaDB | Local, persistent, no external dependencies |
| Query expansion | HyDE (Hypothetical Document Embeddings) | Bridges gap between question phrasing and document language |
| Query expansion | Multi-Query retrieval | Generates alternative phrasings to widen candidate pool and fix hard retrieval misses |
| Reranking | CrossEncoder (`ms-marco-MiniLM-L-6-v2`) | Precision boost over embedding similarity alone |
| Retrieval strategy | MMR (Maximal Marginal Relevance) | Reduces redundant chunks in retrieved context |
| Generation | Claude API (`claude-sonnet-4-6`) | Instruction-following, citation-grounded answers |
| UI | Streamlit | ML-standard, rapid deployment |
| Evaluation | RAGAS | Retrieval + answer quality metrics |
| Deployment | HuggingFace Spaces | Free, recognized in the ML community |

---

## Evaluation (RAGAS)

Evaluated on a manually curated set of 10 question/answer pairs drawn from the ingested papers.
Four configurations compared — each adding one optimization on top of the previous:

| Metric | Baseline (800) | 1400 chunks | + HyDE | + Multi-Query | What it measures |
|--------|---------------|-------------|--------|---------------|-----------------|
| Faithfulness | 0.854 | 0.931 | 0.915 | **0.928** | Are claims in the answer grounded in retrieved context? |
| Answer Relevancy | 0.677 | 0.729 | 0.830 | **0.845** | Does the answer address the question? |
| Context Precision | 0.747 | 0.647 | 0.731 | 0.652 | Are retrieved chunks relevant to the question? |
| Semantic Similarity | 0.824 | 0.811 | 0.835 | **0.838** | How close is the answer to the ground truth? |
| Context Recall | 0.400 | 0.450 | 0.500 | **0.650** | Did retrieval cover all facts in the ground truth? |

**Optimizations applied:**
- **Chunk size 800 → 1400 chars** — splits at paragraph boundaries instead of mid-sentence, improving chunk coherence and faithfulness
- **HyDE (Hypothetical Document Embeddings)** — Claude generates a plausible answer before retrieval; its embedding aligns better with paper language than the raw question, improving recall and answer relevancy
- **Multi-Query retrieval** — Claude generates 2 alternative phrasings of each question; candidates from all 3 queries are merged and reranked together, fixing hard retrieval misses where the original phrasing embeds far from the relevant chunks (context recall +15%)
- **Domain-augmented fallback** — when HyDE + CrossEncoder returns "I don't know", the pipeline appends domain terms ("spiking neural network thesis") to the question and retries with plain MMR; this shifts the embedding toward the thesis's own structural chunks (ToC, section headers) which contain those terms, fixing structural questions like "what models were developed?"

---

## Project Structure

```
snn-research-assistant/
├── app.py                  ← Streamlit UI (entry point)
├── requirements.txt        ← Runtime dependencies
├── requirements-eval.txt   ← Evaluation-only dependencies
├── src/
│   ├── ingest.py           ← PDF loading, chunking, embedding, storing in Chroma
│   ├── retriever.py        ← Multi-Query + HyDE + MMR retrieval + CrossEncoder reranking
│   ├── generator.py        ← Claude API call with retrieved context + citations
│   ├── pipeline.py         ← ask(question) → {answer, sources}
│   └── evaluate.py         ← RAGAS evaluation runner (--hyde, --multi_query flags)
├── chroma_db/              ← Pre-built vector store (committed, ready to use)
├── evaluation_results_baseline_800.json  ← RAGAS results — baseline (800-char chunks)
├── evaluation_results_1400.json          ← RAGAS results — 1400-char chunks
├── evaluation_results_1400_hyde.json     ← RAGAS results — 1400 chunks + HyDE
└── evaluation_results_1400_hyde_mq.json  ← RAGAS results — 1400 chunks + HyDE + Multi-Query (best)
```

---

## How to Run Locally

```bash
git clone https://github.com/rotemso23/snn-research-assistant
cd snn-research-assistant

python -m venv venv
venv/Scripts/python -m pip install -r requirements.txt   # Windows
# source venv/bin/activate && pip install -r requirements.txt  # Mac/Linux

cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

venv/Scripts/streamlit run app.py
```

The pre-built ChromaDB is included — no re-ingestion needed. The app is ready immediately after install.

To add your own papers:
```bash
# Place PDFs in papers/
venv/Scripts/python src/ingest.py --papers_dir papers/
```

---

## Background

Built as an M.Sc. portfolio project (Biomedical Engineering, Technion, 2026) to demonstrate end-to-end ML system ownership:
ingestion → embedding → retrieval → reranking → generation → evaluation → deployment.

The ingested papers cover Spiking Neural Networks, neuromorphic chips, and low-power deep learning — the domain of my thesis on bio-inspired neural computation.
