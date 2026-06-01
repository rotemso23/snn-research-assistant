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

The pipeline is orchestrated as a **LangGraph StateGraph** with LLM-driven nodes — the graph topology (edges and routing conditions) is fixed in Python, but each node's decision is made by Claude Haiku: classifying query intent, decomposing multi-part questions, grading retrieved chunks, and deciding whether to retry with a more targeted query.

```
                          START
                            │
                            ▼
              ┌─────────────────────────┐
              │       router_node       │  Claude Haiku classifies query intent:
              │                         │  simple_factual / thesis_specific /
              │                         │  conceptual_snn / multi_part
              └────────────┬────────────┘
                           │
              ┌────────────┴────────────┐
              │                         │
         multi_part               all other types
              │                         │
              ▼                         │
  ┌───────────────────────┐             │
  │     decompose_node    │  Haiku splits into 2–3 independent sub-queries
  └───────────┬───────────┘             │
              │                         │
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │     retrieve_rerank     │  HyDE + Multi-Query + MMR + CrossEncoder
              └────────────┬────────────┘  (fan-out per sub-query if decomposed)
                           │
                           ▼
              ┌─────────────────────────┐
              │       grade_node        │  Claude Haiku selects relevant chunks
              │                         │  and decides: needs_more_context? (bool)
              │                         │  If yes → produces missing_aspects query
              └────────────┬────────────┘
                           │
              ┌────────────┴────────────┐
              │                         │
      needs_more=True             needs_more=False
      retry_count < 2             (or budget exhausted)
              │                         │
              ▼                         ▼
     [retrieve_rerank]          ┌─────────────────────────┐
     (missing_aspects used as   │      answer_node        │  Claude Sonnet — answers
      next retrieval query)     │                         │  from context only
              │                 └────────────┬────────────┘
              └──► [grade_node]             END
```

**Inside `retrieve_rerank` (standard path):**
```
Multi-Query expansion  →  HyDE expansion  →  MMR retrieval (fetch_k=35 × 3 queries)
      │                                              │
      └──────────────── merge candidates ───────────┘
                                │
                        Hebrew chunk filter     (drops chunks where >20% of letters are Hebrew)
                                │
                        Thesis pool expansion   (router's thesis_specific label adds thesis-source
                                                 candidates before CrossEncoder — no keyword matching)
                                │
                        CrossEncoder rerank     (ms-marco-MiniLM-L-6-v2, top_k=10)
```

**Inside `retrieve_rerank` (fan-out path — multi-part questions only):**
```
sub-query 1 ──► retrieve_and_rerank() ──┐
sub-query 2 ──► retrieve_and_rerank() ──┼──► deduplicate merged pool
sub-query 3 ──► retrieve_and_rerank() ──┘         │
                                                   ▼
                                    CrossEncoder re-rank against original question → top_k=10
```

> Graph diagrams: [`agent_graph.png`](agent_graph.png) — agentic RAG pipeline · [`crag_graph.png`](crag_graph.png) — previous pipeline (document grading)

---

## Tech Stack

| Layer | Tool | Why |
|-------|------|-----|
| Graph | LangGraph `StateGraph` | Stateful graph with conditional edges — LLM-driven nodes for routing, grading, and decomposition |
| Embeddings | `BAAI/bge-large-en-v1.5` (HuggingFace) | State-of-the-art retrieval embeddings |
| Vector store | ChromaDB | Local, persistent, no external dependencies |
| Query routing | Claude Haiku (`claude-haiku-4-5`) | Classifies query intent (4 types) to adjust retrieval strategy before any retrieval happens |
| Query decomposition | Claude Haiku (`claude-haiku-4-5`) | Splits multi-part questions into 2–3 independent sub-queries for fan-out retrieval |
| Query expansion | HyDE (Hypothetical Document Embeddings) | Bridges gap between question phrasing and document language |
| Query expansion | Multi-Query retrieval | Generates alternative phrasings to widen candidate pool and fix hard retrieval misses |
| Document grading | Claude Haiku (`claude-haiku-4-5`) | Grades all chunks in one call; decides `needs_more_context` (bool) and produces `missing_aspects` as the next retrieval query |
| Reranking | CrossEncoder (`ms-marco-MiniLM-L-6-v2`) | Precision boost over embedding similarity alone |
| Retrieval strategy | MMR (Maximal Marginal Relevance) | Reduces redundant chunks in retrieved context |
| Generation | Claude API (`claude-sonnet-4-6`) | Instruction-following, citation-grounded answers |
| UI | Streamlit | ML-standard, rapid deployment |
| Evaluation | RAGAS | Retrieval + answer quality metrics |
| Deployment | HuggingFace Spaces | Free, recognized in the ML community |

---

## Evaluation (RAGAS)

Evaluated on a manually curated set of 10 question/answer pairs drawn from the ingested papers.
Five configurations compared — each adding one optimization on top of the previous:

| Metric | Baseline (800) | 1400 chunks | + HyDE | + Multi-Query | + Grading | **Agentic RAG** | What it measures |
|--------|:--------------:|:-----------:|:------:|:-------------:|:---------:|:---------------:|:----------------|
| Faithfulness | 0.854 | 0.931 | 0.915 | 0.928 | 0.909 | **0.981** | Are claims in the answer grounded in retrieved context? |
| Answer Relevancy | 0.677 | 0.729 | 0.830 | 0.845 | 0.946 | **0.947** | Does the answer address the question? |
| Context Precision | 0.747 | 0.647 | 0.731 | 0.652 | 0.780 | **0.870** | Are retrieved chunks relevant to the question? |
| Context Recall | 0.400 | 0.450 | 0.500 | 0.650 | 0.650 | **0.800** | Did retrieval cover all facts in the ground truth? |
| Semantic Similarity | 0.824 | 0.811 | 0.835 | 0.838 | 0.859 | **0.864** | How close is the answer to the ground truth? |

**Optimizations applied (cumulative):**
- **Chunk size 800 → 1400 chars** — splits at paragraph boundaries instead of mid-sentence, improving chunk coherence and faithfulness
- **HyDE (Hypothetical Document Embeddings)** — Claude generates a plausible answer before retrieval; its embedding aligns better with paper language than the raw question, improving recall and answer relevancy
- **Multi-Query retrieval** — Claude generates 2 alternative phrasings of each question; candidates from all 3 queries are merged and reranked together, fixing hard retrieval misses where the original phrasing embeds far from the relevant chunks (context recall +15%)
- **Hebrew chunk filtering** — the thesis includes a Hebrew abstract; the English-only embedding model (`bge-large-en-v1.5`) embeds Hebrew text poorly, causing it to surface irrelevantly. Chunks where >20% of letters are Hebrew are dropped post-retrieval, removing noise without affecting English content
- **LangGraph grading (document relevance filter)** — Claude Haiku grades each retrieved chunk against the question before generation; irrelevant chunks are discarded and the query is rewritten if all chunks fail. Removing noise from the context significantly boosts answer relevancy (+10%) and context precision (+13%)
- **Agentic RAG** — LLM-driven routing, decomposition, and grading replacing fixed rewrite/fallback nodes:
  - **Query routing** — Haiku classifies each question before retrieval; `simple_factual` skips HyDE/multi-query, `thesis_specific` activates thesis pool expansion regardless of phrasing
  - **Multi-part decomposition** — compound questions split into 2–3 sub-queries; each retrieved independently, merged pool re-ranked by CrossEncoder against the original question
  - **Router-driven thesis boost** — thesis expansion now triggered by router classification, not keyword matching
  - **Agentic grading** — single Haiku call grades all chunks, decides `needs_more_context` (bool), and produces `missing_aspects` as the next retrieval query; replaces the separate `rewrite_query` node
  - **Wider funnel** — `fetch_k` 20 → 35, `top_k` 7 → 10

---

## Project Structure

```
snn-research-assistant/
├── app.py                  ← Streamlit UI (entry point)
├── requirements.txt        ← Runtime dependencies
├── requirements-eval.txt   ← Evaluation-only dependencies
├── src/
│   ├── ingest.py           ← PDF loading, chunking, embedding, storing in Chroma
│   ├── retriever.py        ← Multi-Query + HyDE + MMR + Hebrew filter + thesis boost + CrossEncoder + fan-out retrieval
│   ├── generator.py        ← Claude API call with retrieved context + citations
│   ├── agent_graph.py      ← Agentic LangGraph StateGraph (router, decompose, retrieve_rerank, grade, answer)
│   ├── graph.py            ← Original CRAG pipeline (kept for reference and --grading eval)
│   ├── pipeline.py         ← ask(question) → {answer, sources}  (thin wrapper over agent_graph.py)
│   └── evaluate.py         ← RAGAS evaluation runner (--hyde, --multi_query, --agentic flags)
├── chroma_db/              ← Pre-built vector store (committed, ready to use)
├── evaluation_results_baseline_800.json    ← RAGAS results — baseline (800-char chunks)
├── evaluation_results_1400.json            ← RAGAS results — 1400-char chunks
├── evaluation_results_1400_hyde.json       ← RAGAS results — 1400 chunks + HyDE
├── evaluation_results_1400_hyde_mq.json    ← RAGAS results — 1400 chunks + HyDE + Multi-Query
├── evaluation_results_grading.json         ← RAGAS results — original CRAG pipeline + grading
└── evaluation_results_agentic.json         ← RAGAS results — agentic RAG (best)
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

To run the RAGAS evaluation:
```bash
venv/Scripts/python -m pip install -r requirements-eval.txt
venv/Scripts/python src/evaluate.py --agentic --hyde --multi_query   # agentic RAG (best config)
venv/Scripts/python src/evaluate.py --grading --hyde --multi_query   # original CRAG pipeline
venv/Scripts/python src/evaluate.py                                   # baseline (no HyDE, no multi-query)
```

---

## Background

Built as an M.Sc. portfolio project (Biomedical Engineering, Technion, 2026) to demonstrate end-to-end ML system ownership:
ingestion → embedding → retrieval → reranking → generation → evaluation → deployment.

The ingested papers cover Spiking Neural Networks, neuromorphic chips, and low-power deep learning — the domain of my thesis on bio-inspired neural computation.
