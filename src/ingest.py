"""
ingest.py — Load PDFs, chunk, embed, and store in ChromaDB.

Usage:
    python src/ingest.py --papers_dir papers/
"""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

CHROMA_DIR = "chroma_db"
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200


def load_pdfs(papers_dir: str) -> list:
    """Load all PDFs from a directory and return LangChain Document objects."""
    papers_path = Path(papers_dir)
    pdf_files = list(papers_path.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {papers_dir}")

    all_docs = []
    for pdf_path in pdf_files:
        print(f"Loading: {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        all_docs.extend(docs)

    print(f"Loaded {len(all_docs)} pages from {len(pdf_files)} PDF(s)")
    return all_docs


def chunk_documents(docs: list) -> list:
    """Split documents into chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")
    return chunks


def embed_and_store(chunks: list) -> Chroma:
    """Embed chunks and store in ChromaDB."""
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print(f"Storing {len(chunks)} chunks in ChromaDB at '{CHROMA_DIR}'...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    print(f"Done. ChromaDB persisted at '{CHROMA_DIR}'")
    return vectorstore


def ingest(papers_dir: str) -> int:
    """Full ingestion pipeline. Returns total chunk count."""
    docs = load_pdfs(papers_dir)
    chunks = chunk_documents(docs)
    embed_and_store(chunks)
    return len(chunks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDFs into ChromaDB")
    parser.add_argument(
        "--papers_dir",
        type=str,
        default="papers/",
        help="Directory containing PDF files",
    )
    args = parser.parse_args()

    total = ingest(args.papers_dir)
    print(f"\nIngestion complete. Total chunks stored: {total}")
