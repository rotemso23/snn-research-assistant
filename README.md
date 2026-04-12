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

A RAG (Retrieval-Augmented Generation) pipeline for exploring academic papers on
Spiking Neural Networks, neuromorphic computing, and low-power deep learning.

**Stack:** LangChain · ChromaDB · HuggingFace sentence-transformers · Claude API · Streamlit

## What it does

Ask a natural-language question about SNNs and the assistant retrieves the most relevant
chunks from a curated set of papers, then generates a grounded answer with source citations.
It will say "I don't know" if the context doesn't support an answer.

## Evaluation (RAGAS)

Evaluated on a manually curated set of 10 question/answer pairs from the ingested papers:

| Metric | Score |
|--------|-------|
| Faithfulness | 0.854 |
| Answer Relevancy | 0.677 |
| Context Precision | 0.747 |
| Semantic Similarity | 0.824 |

*Context recall (0.400) is lower because several ground-truth answers are broad thesis-level
claims requiring more context than the top-7 retrieved chunks can cover — a known limitation.*

## How to run locally

```bash
git clone <repo-url>
cd snn-research-assistant
python -m venv venv
venv/Scripts/python -m pip install -r requirements.txt
cp .env.example .env   # add your ANTHROPIC_API_KEY
venv/Scripts/streamlit run app.py
```

Built as an M.Sc. portfolio project (Biomedical Engineering, Technion, 2026).
