#!/usr/bin/env python3
"""
Quickstart Demo — Hybrid RAG retrieval without models.

This script demonstrates the deterministic core of the Hybrid RAG pipeline:
  1. Load a small cybersecurity corpus (.txt files)
  2. Chunk the text using TextChunker (semantic paragraph-based splitting)
  3. Build a BM25 keyword index from the chunks
  4. Run example queries and retrieve the top-ranked chunks

NO embeddings, NO ChromaDB, NO Ollama, NO GPU required.
Runs in <5 seconds on CPU.

Usage:
    python examples/quickstart/run_demo.py
"""
import re
import sys
import time
from pathlib import Path

# Force UTF-8 stdout (Windows console defaults to cp1252 which breaks em-dashes)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure project root and src/ are on sys.path (works without editable install)
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for p in (str(ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from chunker import TextChunker
from rank_bm25 import BM25Okapi


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def load_corpus(documents_dir: Path) -> list[dict]:
    """Load all .txt files from the documents directory.

    Returns a list of dicts with keys: filename, text.
    """
    docs = []
    for txt_path in sorted(documents_dir.glob("*.txt")):
        text = txt_path.read_text(encoding="utf-8").strip()
        if text:
            docs.append({
                "filename": txt_path.name,
                "text": text,
            })
    return docs


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_corpus(docs: list[dict], chunk_size: int = 800, overlap: int = 200) -> list[dict]:
    """Chunk the corpus using TextChunker and attach metadata.

    Returns a list of chunk dicts with keys: id, text, metadata.
    """
    chunker = TextChunker(chunk_size=chunk_size, overlap=overlap)
    all_chunks = []
    chunk_id = 0
    for doc in docs:
        chunks = chunker.split_text_semantic(doc["text"])
        for i, chunk_text in enumerate(chunks):
            all_chunks.append({
                "id": f"{doc['filename']}_chunk_{i}",
                "text": chunk_text,
                "metadata": {
                    "source": doc["filename"],
                    "chunk_index": chunk_id,
                },
            })
            chunk_id += 1
    return all_chunks


# ---------------------------------------------------------------------------
# BM25 search
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    """Simple tokenizer: lowercase, split on non-alphanumeric."""
    return [t for t in re.split(r"\W+", text.lower()) if t]


def build_bm25_index(chunks: list[dict]) -> BM25Okapi:
    """Build a BM25 index from the chunk texts."""
    tokenized = [tokenize(c["text"]) for c in chunks]
    return BM25Okapi(tokenized)


def search(query: str, chunks: list[dict], bm25: BM25Okapi, top_k: int = 3) -> list[dict]:
    """Search the BM25 index and return the top-k chunks with scores."""
    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    results = []
    for idx, score in ranked[:top_k]:
        if score <= 0:
            break
        results.append({
            "rank": len(results) + 1,
            "score": round(score, 4),
            "source": chunks[idx]["metadata"]["source"],
            "text": chunks[idx]["text"][:300] + ("..." if len(chunks[idx]["text"]) > 300 else ""),
        })
    return results


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

EXAMPLE_QUERIES = [
    "What are the core functions of the NIST Cybersecurity Framework?",
    "How many controls does ISO 27001 define?",
    "What is a next-generation firewall?",
    "What are the eight domains of the CISSP certification?",
    "What is the core principle of Zero Trust?",
]


def main():
    print("=" * 72)
    print("  Hybrid RAG Quickstart — Deterministic Retrieval Demo (no models)")
    print("=" * 72)
    print()

    # 1. Load corpus
    documents_dir = Path(__file__).parent / "documents"
    docs = load_corpus(documents_dir)
    print(f"[1/3] Loaded {len(docs)} documents from {documents_dir.name}/")
    for d in docs:
        print(f"       - {d['filename']} ({len(d['text'])} chars)")
    print()

    # 2. Chunk
    t0 = time.perf_counter()
    chunks = chunk_corpus(docs)
    t_chunk = time.perf_counter() - t0
    print(f"[2/3] Chunked into {len(chunks)} chunks (chunk_size=800, overlap=200)")
    print(f"       Chunking time: {t_chunk*1000:.1f} ms")
    print()

    # 3. Build BM25 index
    t0 = time.perf_counter()
    bm25 = build_bm25_index(chunks)
    t_index = time.perf_counter() - t0
    print(f"[3/3] Built BM25 index over {len(chunks)} chunks")
    print(f"       Indexing time: {t_index*1000:.1f} ms")
    print()

    # 4. Run example queries
    print("-" * 72)
    print("  Example Queries")
    print("-" * 72)
    total_search_time = 0.0
    for query in EXAMPLE_QUERIES:
        t0 = time.perf_counter()
        results = search(query, chunks, bm25, top_k=3)
        t_search = time.perf_counter() - t0
        total_search_time += t_search

        print(f"\n  Query: \"{query}\"")
        print(f"  Search time: {t_search*1000:.2f} ms")
        if not results:
            print("  No matching chunks found.")
            continue
        for r in results:
            print(f"  #{r['rank']} [score={r['score']}] {r['source']}")
            # Print first 120 chars of text, single line
            preview = r["text"].replace("\n", " ")[:120]
            print(f"      {preview}...")
    print()
    print("-" * 72)
    print(f"  Total search time for {len(EXAMPLE_QUERIES)} queries: {total_search_time*1000:.1f} ms")
    print(f"  Total demo time: {(t_chunk + t_index + total_search_time)*1000:.1f} ms")
    print()
    print("  This demo used ONLY TextChunker + BM25 (no embeddings, no ChromaDB,")
    print("  no Ollama). The full Hybrid RAG pipeline adds semantic embeddings,")
    print("  vector search, reciprocal rank fusion, and LLM generation.")
    print("=" * 72)


if __name__ == "__main__":
    main()
