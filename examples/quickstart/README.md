# Quickstart — 5-Minute Demo (No Models Required)

This demo runs the **deterministic core** of the Hybrid RAG pipeline: text chunking +
BM25 keyword retrieval. It requires **no embeddings, no ChromaDB, no Ollama, no GPU**.

## Prerequisites

```powershell
pip install -e ".[dev]"
```

## Run

```powershell
python examples/quickstart/run_demo.py
```

## What it does

1. **Loads** 5 cybersecurity `.txt` documents from `documents/` (~2 KB each)
2. **Chunks** them using `TextChunker` (semantic paragraph-based splitting, 800 chars, 200 overlap)
3. **Indexes** the chunks with BM25 (`rank_bm25`)
4. **Runs** 5 example queries and prints the top-3 ranked chunks with BM25 scores

## Expected output

```
[1/3] Loaded 5 documents from documents/
[2/3] Chunked into 19 chunks (chunk_size=800, overlap=200)
[3/3] Built BM25 index over 19 chunks

Query: "What are the core functions of the NIST Cybersecurity Framework?"
#1 [score=14.04] 01_nist_cybersecurity_framework.txt
#2 [score=10.01] 01_nist_cybersecurity_framework.txt
#3 [score=7.81]  05_zero_trust_architecture.txt

Total demo time: ~4 ms
```

## What this demonstrates

- **Chunking**: `TextChunker.split_text_semantic` respects paragraph and sentence boundaries.
- **BM25 retrieval**: Keyword-based ranking returns relevant chunks for domain-specific queries.
- **Determinism**: Same input always produces the same output — no randomness, no model variance.

## What the full pipeline adds

The complete Hybrid RAG system extends this with:
- **Semantic embeddings** (BGE-M3 via sentence-transformers or nomic-embed-text via Ollama)
- **Vector search** (ChromaDB with HNSW cosine similarity)
- **Hybrid fusion** (weighted blend of semantic + BM25 scores)
- **LLM generation** (Ollama with citation-grounded prompting)
- **Factual gating** (deterministic pre-LLM evidence checks)

These require local models and are documented in the main [README](../../README.md).

## Corpus

The 5 documents cover:

| File | Topic |
|---|---|
| `01_nist_cybersecurity_framework.txt` | NIST CSF 2.0 — six core functions, tiers, profiles |
| `02_iso_27001_controls.txt` | ISO 27001:2022 — 93 controls, 4 themes, PDCA cycle |
| `03_firewall_fundamentals.txt` | Firewall types, deployment architectures, best practices |
| `04_cissp_certification_domains.txt` | CISSP — eight domains of the CBK, exam format |
| `05_zero_trust_architecture.txt` | Zero Trust — principles, PDP/PEP, implementation pillars |
