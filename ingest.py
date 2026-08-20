"""Extract -> Chunk -> Embed -> Store pipeline for the resume RAG system."""
import os
import argparse

from pypdf import PdfReader
import chromadb
from chromadb.utils import embedding_functions

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PDF = os.path.join(BASE_DIR, "data", "resume.pdf")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "resume"


def extract_text(pdf_path):
    reader = PdfReader(pdf_path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def chunk_text(text, chunk_size=500, overlap=80):
    """Split into paragraph-aware chunks of ~chunk_size chars with overlap."""
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    chunks = []
    current = ""
    for para in paragraphs:
        if current and len(current) + len(para) + 1 > chunk_size:
            chunks.append(current)
            current = current[-overlap:] + "\n" + para
        else:
            current = f"{current}\n{para}" if current else para
    if current:
        chunks.append(current)
    return chunks


def build_collection(pdf_path=DEFAULT_PDF, reset=False):
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    ef = embedding_functions.DefaultEmbeddingFunction()

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    collection = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=ef)

    text = extract_text(pdf_path)
    chunks = chunk_text(text)

    ids = [f"chunk-{i}" for i in range(len(chunks))]
    metadatas = [{"source": os.path.basename(pdf_path), "chunk_index": i} for i in range(len(chunks))]

    collection.upsert(documents=chunks, ids=ids, metadatas=metadatas)
    return collection, len(chunks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest a resume PDF into ChromaDB")
    parser.add_argument("--pdf", default=DEFAULT_PDF, help="Path to the resume PDF")
    parser.add_argument("--reset", action="store_true", help="Drop and rebuild the collection")
    args = parser.parse_args()

    _, n = build_collection(args.pdf, reset=args.reset)
    print(f"Ingested {n} chunks from {args.pdf} into ChromaDB at {CHROMA_PATH}")
