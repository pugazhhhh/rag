"""Retrieve -> Answer pipeline: embed the question, pull top-k chunks from
ChromaDB, and pass them to an LLM (or print them directly if no LLM is
configured) to answer questions about the resume.
"""
import os
import sys
import time
import argparse

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

from ingest import CHROMA_PATH, COLLECTION_NAME, build_collection, DEFAULT_PDF

load_dotenv()

SYSTEM_PROMPT = (
    "You are a resume assistant. Answer the user's question using ONLY the "
    "resume excerpts provided as context. If the answer isn't in the "
    "context, say you don't have that information."
)


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    ef = embedding_functions.DefaultEmbeddingFunction()
    try:
        return client.get_collection(COLLECTION_NAME, embedding_function=ef)
    except Exception:
        collection, _ = build_collection(DEFAULT_PDF)
        return collection


def retrieve(question, n_results=3):
    collection = get_collection()
    results = collection.query(query_texts=[question], n_results=n_results)
    return results["documents"][0], results["distances"][0]


def generate_answer(question, chunks):
    context = "\n\n---\n\n".join(chunks)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return (
            "[No GEMINI_API_KEY set - showing retrieved context instead of an LLM answer]\n\n"
            + context
        )

    from google import genai
    from google.genai import types
    from google.genai.errors import ServerError

    client = genai.Client(api_key=api_key)

    attempts = 3
    for attempt in range(1, attempts + 1):
        try:
            response = client.models.generate_content(
                model="gemini-flash-latest",
                contents=f"Context:\n{context}\n\nQuestion: {question}",
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0,
                    max_output_tokens=1000,
                ),
            )
            return response.text
        except ServerError as e:
            if attempt == attempts:
                return (
                    f"[Gemini is unavailable after {attempts} attempts ({e}) - "
                    "showing retrieved context instead]\n\n" + context
                )
            time.sleep(2 ** attempt)


def answer(question, n_results=3, verbose=False):
    chunks, distances = retrieve(question, n_results)
    if verbose:
        print("Retrieved chunks:")
        for c, d in zip(chunks, distances):
            print(f"  (distance={d:.4f}) {c[:80]}...")
        print()
    return generate_answer(question, chunks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ask questions about the resume")
    parser.add_argument("question", nargs="*", help="Question to ask (omit for interactive mode)")
    parser.add_argument("-k", type=int, default=3, help="Number of chunks to retrieve")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show retrieved chunks")
    args = parser.parse_args()

    if args.question:
        print(answer(" ".join(args.question), n_results=args.k, verbose=args.verbose))
    else:
        print("Resume RAG - type a question (or 'exit' to quit)")
        while True:
            try:
                q = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q or q.lower() in ("exit", "quit"):
                break
            print(answer(q, n_results=args.k, verbose=args.verbose))
