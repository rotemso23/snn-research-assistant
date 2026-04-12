# CLAUDE.md — SNN Research Assistant

## What this project is

A RAG pipeline that lets users ask questions over a collection of academic papers on
Spiking Neural Networks (SNNs) and neuromorphic computing. Built with LangChain,
ChromaDB, HuggingFace sentence-transformers, and the Claude API. Deployed on
HuggingFace Spaces with a Streamlit UI.

## Tech stack

| Layer | Tool |
|-------|------|
| Orchestration | LangChain |
| Embeddings | `BAAI/bge-large-en-v1.5` (HuggingFace) |
| Vector store | ChromaDB (`chroma_db/`) |
| Reranking | CrossEncoder `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Generation | Claude API (`claude-sonnet-4-6`) |
| UI | Streamlit (`app.py`) |
| Evaluation | RAGAS (`src/evaluate.py`) |

## Project structure

```
snn-research-assistant/
├── app.py                  ← Streamlit entry point
├── requirements.txt        ← Runtime dependencies
├── requirements-eval.txt   ← Evaluation-only dependencies (ragas, rouge-score, nltk)
├── src/
│   ├── ingest.py           ← PDF loading, chunking, embedding, storing in Chroma
│   ├── retriever.py        ← MMR retrieval + CrossEncoder reranking
│   ├── generator.py        ← Claude API call with context + citations
│   ├── pipeline.py         ← ask(question: str) -> {"answer": str, "sources": list[str]}
│   └── evaluate.py         ← RAGAS evaluation runner
├── chroma_db/              ← Pre-built vector store (committed via git LFS)
└── evaluation_results.json ← RAGAS results for 10 test questions
```

## Pipeline configuration

| Setting | Value |
|---------|-------|
| Chunk size | 800 |
| Chunk overlap | 200 |
| Retrieval | MMR, fetch_k=15 |
| Reranking | CrossEncoder, top_k=7 |
| Generation | max_tokens=1024, answers only from provided context |

## Running locally

```bash
python -m venv venv
venv/Scripts/python -m pip install -r requirements.txt
cp .env.example .env   # add ANTHROPIC_API_KEY
venv/Scripts/streamlit run app.py
```

Always use `venv/Scripts/python`, not the system Python.

## Adding new papers

```bash
# Place PDFs in papers/
venv/Scripts/python src/ingest.py --papers_dir papers/
```

This re-builds `chroma_db/`. Commit the updated `chroma_db/` and push to both remotes.

## Git remotes

- `origin` → GitHub: https://github.com/rotemso23/snn-research-assistant
- `space` → HuggingFace: https://huggingface.co/spaces/rotemso23/snn-research-assistant

`papers/` and `.env` are gitignored on both remotes.
`chroma_db/` is committed via git LFS (`*.sqlite3` and `*.bin` tracked).

## Environment variables

- `ANTHROPIC_API_KEY` — required. Set in `.env` locally, Space secret on HuggingFace.
