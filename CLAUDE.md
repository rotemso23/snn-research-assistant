# CLAUDE.md — SNN Research Assistant (RAG Project)

## What this project is

A RAG (Retrieval-Augmented Generation) pipeline that lets users ask questions over a collection of academic papers on Spiking Neural Networks (SNNs), neuromorphic computing, and low-power deep learning — the domain of Rotem's M.Sc. thesis at the Technion.

This is a CV portfolio project. The goal is a deployed, working product on HuggingFace Spaces with a clean GitHub repo.

## Motivation / context

Rotem is a fresh M.Sc. graduate (Biomedical Engineering, Technion, 2026) seeking AI/ML roles in Israel. Recurring gaps across CV analyses: RAG, HuggingFace ecosystem, LangChain, embeddings, vector databases, cloud deployment, and NLP domain experience. This project closes all of them in one shot.

## Tech stack

| Layer | Tool | Why |
|-------|------|-----|
| Orchestration | LangChain | Closes LangChain gap; appears in many JD analyses |
| Embeddings | HuggingFace sentence-transformers (`all-MiniLM-L6-v2` or similar) | Closes HuggingFace gap |
| Vector store | ChromaDB | Simple, local, no account needed |
| Generation | Claude API (Anthropic) | Powerful, pay-per-use — needs account + API key at console.anthropic.com |
| UI | Streamlit | Simple, ML-standard |
| Evaluation | RAGAS | Retrieval + answer quality metrics |
| Deployment | HuggingFace Spaces | Free, recognized in ML community |

## Project structure

```
snn-research-assistant/
├── CLAUDE.md               ← this file
├── README.md               ← write last, CV-quality
├── requirements.txt
├── .env.example            ← ANTHROPIC_API_KEY placeholder (never commit real keys)
├── .gitignore
├── papers/                 ← PDF papers go here (not committed to git — add to .gitignore)
├── chroma_db/              ← persisted vector store (not committed to git)
├── src/
│   ├── ingest.py           ← PDF loading, chunking, embedding, storing in Chroma
│   ├── retriever.py        ← query pipeline: embed question → retrieve chunks
│   ├── generator.py        ← Claude API call with retrieved context + citations
│   ├── pipeline.py         ← connects retriever + generator end-to-end
│   └── evaluate.py         ← RAGAS evaluation
├── app.py                  ← Streamlit UI (entry point)
└── notebooks/
    └── exploration.ipynb   ← optional: chunk size experiments, embedding comparisons
```

## Build phases — do these in order

### Phase 1: Ingestion pipeline (`src/ingest.py`)
- Load PDFs from `papers/` using `langchain_community.document_loaders.PyPDFLoader`
- Chunk with `RecursiveCharacterTextSplitter` (chunk_size=800, chunk_overlap=100 — these are starting defaults, tune later)
- Embed with HuggingFace `sentence-transformers/all-MiniLM-L6-v2`
- Store in ChromaDB with persistence so it survives restarts
- Add a CLI: `python src/ingest.py --papers_dir papers/` that prints chunk count on completion

### Phase 2: Retrieval + generation pipeline (`src/retriever.py`, `src/generator.py`, `src/pipeline.py`)
- Retriever: embed the query, pull top-k chunks (k=5 default) from Chroma
- Generator: send retrieved chunks to Claude as context, with a system prompt that:
  - Instructs Claude to answer only from the provided context
  - Instructs Claude to cite the source paper for each claim (by filename or title)
  - Instructs Claude to say "I don't know" if the context doesn't contain the answer
- Pipeline: single `ask(question: str) -> dict` function returning `{"answer": str, "sources": list[str]}`

### Phase 3: Streamlit UI (`app.py`)
- Text input for the question
- Answer displayed with sources listed below
- Keep it clean and simple — this is a demo, not a product

### Phase 4: Evaluation (`src/evaluate.py`)
- Create a small test set: 10–15 question/answer pairs written manually from the papers
- Run RAGAS metrics: `faithfulness`, `answer_relevancy`, `context_recall`
- Save results to `evaluation_results.json`
- These numbers go in the README

### Phase 5: Deployment (HuggingFace Spaces)
- Create a HuggingFace account if not already done
- Create a new Space: Streamlit SDK
- Add `ANTHROPIC_API_KEY` as a Space secret (never hardcoded)
- The vector DB needs to either: (a) be committed to the repo pre-built, or (b) be rebuilt on Space startup from papers committed to the repo
- Recommended: commit the pre-built ChromaDB directory and the papers to a private HF dataset or directly to the Space repo (if papers are not too large)

### Phase 6: README + GitHub
- README must include: what it does, tech stack, architecture diagram (even a simple text one), evaluation results, how to run locally, link to live demo
- This README is what a recruiter will read — write it to show you understand the system, not just that you built it

## Key conventions

- All secrets via environment variables (`.env` file locally, Space secrets in HF). Never hardcoded.
- `papers/` and `chroma_db/` in `.gitignore`
- Type hints on all functions
- Each module has a `if __name__ == "__main__"` block for standalone testing
- Git commit after each phase is complete

## Honest gaps this project closes (for CV)

After completing this project, Rotem can honestly claim:
- RAG pipeline design and implementation
- Embeddings and vector search (ChromaDB)
- HuggingFace sentence-transformers
- LangChain orchestration
- Claude API integration
- Streamlit application development
- RAGAS evaluation methodology
- HuggingFace Spaces deployment
- End-to-end ML system ownership (ingestion → retrieval → generation → evaluation → deployment)

## What NOT to overclaim on the CV

- This is not a production system serving real users — describe it as a "deployed research assistant" or "portfolio project"
- ChromaDB is a local vector store, not a managed vector DB (Pinecone, Weaviate) — do not claim managed vector DB experience
- The evaluation set is small and manually created — describe it as a "curated evaluation set" not a benchmark
