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
│  HuggingFace        │   BAAI/bge-large-en-v1.5
│  Embeddings         │   (query embedding)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  ChromaDB           │   MMR retrieval
│  Vector Store       │   fetch_k=15 candidates
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  CrossEncoder       │   ms-marco-MiniLM-L-6-v2
│  Reranker           │   top_k=7 chunks selected
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Claude API         │   claude-sonnet-4-6
│  Generation         │   answers from context only
└────────┬────────────┘
         │
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
| Reranking | CrossEncoder (`ms-marco-MiniLM-L-6-v2`) | Precision boost over embedding similarity alone |
| Retrieval strategy | MMR (Maximal Marginal Relevance) | Reduces redundant chunks in retrieved context |
| Generation | Claude API (`claude-sonnet-4-6`) | Instruction-following, citation-grounded answers |
| UI | Streamlit | ML-standard, rapid deployment |
| Evaluation | RAGAS | Retrieval + answer quality metrics |
| Deployment | HuggingFace Spaces | Free, recognized in the ML community |

---

## Evaluation (RAGAS)

Evaluated on a manually curated set of 10 question/answer pairs drawn from the ingested papers:

| Metric | Score | What it measures |
|--------|-------|-----------------|
| Faithfulness | **0.854** | Are claims in the answer grounded in retrieved context? |
| Answer Relevancy | **0.677** | Does the answer address the question? |
| Context Precision | **0.747** | Are retrieved chunks relevant to the question? |
| Semantic Similarity | **0.824** | How close is the answer to the ground truth? |
| Context Recall | 0.400 | Did retrieval cover all facts in the ground truth? |

> Context recall is lower because several ground-truth answers are broad thesis-level claims
> that span more content than the top-7 retrieved chunks can cover — a known limitation of
> fixed-k retrieval on long-form questions.

---

## Project Structure

```
snn-research-assistant/
├── app.py                  ← Streamlit UI (entry point)
├── requirements.txt        ← Runtime dependencies
├── requirements-eval.txt   ← Evaluation-only dependencies
├── src/
│   ├── ingest.py           ← PDF loading, chunking, embedding, storing in Chroma
│   ├── retriever.py        ← MMR retrieval + CrossEncoder reranking
│   ├── generator.py        ← Claude API call with retrieved context + citations
│   ├── pipeline.py         ← ask(question) → {answer, sources}
│   └── evaluate.py         ← RAGAS evaluation runner
├── chroma_db/              ← Pre-built vector store (committed, ready to use)
└── evaluation_results.json ← Full per-question RAGAS results
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
