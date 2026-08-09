# Hybrid RAG Cybersecurity Knowledge System

![CI](https://github.com/ValenByteTar/SistemaRAGHybrid/actions/workflows/tests.yml/badge.svg)

An offline Hybrid RAG system for cybersecurity knowledge retrieval. It combines semantic search, lexical search (BM25), cross-encoder reranking, and a local LLM to answer questions over a corpus of cybersecurity technical documents — with no external API keys required for the main pipeline.

```
Documents (PDF)
    ↓
Extraction (PyMuPDF, per-page)
    ↓
Chunking (token-based, with page metadata)
    ↓
Embeddings (BGE-M3) ─────┐
                         ├── Hybrid Retrieval (weighted fusion)
BM25 (rank-bm25) ────────┘
                         ↓
                   Reranking (BGE-reranker-v2-m3)
                         ↓
                   Context Building
                         ↓
                   Factual Gate (deterministic)
                         ↓
                   LLM Generation (Ollama, local)
                         ↓
                   Answer Postprocessing (citations, cleanup)
```

---

## What problem does it solve?

Cybersecurity knowledge is scattered across hundreds of PDFs — standards (NIST, ISO 27001, PCI DSS), certification guides (CISSP, CCSP, CEH), vendor documentation, and cheat sheets. Finding a specific fact (e.g. "what are the eight CISSP domains?" or "what does NIST CSF say about Zero Trust?") requires knowing which document to open and which page to read.

This system lets a user ask a natural-language question and retrieves the most relevant chunks from the corpus, ranks them, and generates an answer grounded in the retrieved evidence — with citation markers like `[Doc 3 - NIST CSF v2 p.5]` so the answer is auditable.

The key engineering challenge is **groundedness**: the system must answer when evidence exists and **decline** when it doesn't, rather than hallucinating from the LLM's parametric knowledge. A deterministic factual gate blocks answers to factual questions (CVEs, prices, RFCs, temperatures) when no literal evidence is found in the retrieved context.

---

## Architecture

```
                 +-------------------+
                 |    Web / CLI      |
                 +---------+---------+
                           |
                           v
                 +-------------------+
                 |    HybridRAG      |
                 |   rag_hybrid.py   |
                 +---------+---------+
                           |
          +----------------+----------------+
          |                                 |
          v                                 v
+---------------------+           +---------------------+
| RetrievalEngine     |           | ContextBuilder      |
| - BGE-M3 semantic   |           | - context selection |
| - BM25 lexical      |           | - prompt assembly   |
| - weighted fusion   |           +----------+----------+
| - CrossEncoder      |                      |
+----------+----------+                      v
           |                         +---------------------+
           v                         | Ollama (LLM local)  |
+---------------------+               +----------+----------+
| ChromaDB            |                          |
| BGE-M3 embeddings   |                          v
+---------------------+               +---------------------+
                                     | AnswerPostprocessor |
                                     | + Factual Gate      |
                                     | citations, cleanup  |
                                     +---------------------+
```

### Components

| Component | Responsibility |
|---|---|
| `rag_hybrid.py` | Main orchestrator — coordinates the full query flow |
| `retrieval_engine.py` | Hybrid search: semantic + BM25, weighted fusion, reranking, filtering |
| `context_builder.py` | Context selection and LLM prompt assembly by query type |
| `answer_postprocessor.py` | Response cleanup, citation extraction, source deduplication |
| `query_classifier.py` | Intent classification (out-of-domain, comparison, multi-doc, etc.) |
| `equivalences_manager.py` | Synonym/acronym expansion and query normalization |
| `src/pdf_extractor.py` | Per-page text extraction with PyMuPDF |
| `src/chunker.py` | Token-based or semantic chunking with page/source metadata |
| `src/embedder.py` | Embedding generation (BGE-M3) with caching |
| `src/vector_store.py` | ChromaDB persistence and querying |
| `src/rag/factual_gate.py` | Deterministic gate: blocks factual answers without literal evidence |
| `src/rag/entity_extractor.py` | NER and document-reference extraction from queries |
| `ollama_manager.py` | Ollama lifecycle and availability management |
| `doc_cards.py` | Document role cards for retrieval planning |
| `memory_system.py` | SQLite-based knowledge and conversational memory |

### Hybrid fusion

Retrieval combines two signals into a single score:

```
hybrid_score = semantic_weight * semantic_score + keyword_weight * keyword_score
```

- **Semantic score**: `1 - cosine_distance` from ChromaDB (BGE-M3 embeddings)
- **Keyword score**: BM25 score normalized by the max BM25 score across candidates
- **Default weights**: `semantic_weight = 0.6`, `keyword_weight = 0.4` (configurable in `config.yaml`)

The top-K fused candidates are then reranked by a cross-encoder (`BGE-reranker-v2-m3`), and the final score blends hybrid and rerank scores (50/50 by default).

---

## Dataset

| Property | Value |
|---|---|
| Domain | Cybersecurity (standards, certifications, vendor docs, cheat sheets) |
| Source documents | ~861 PDFs |
| Indexed chunks | 100,480 |
| Chunking | Token-based, 350 tokens per chunk, 50 token overlap |
| Metadata per chunk | Source filename, page number, chunk index, document date |

The corpus is not bundled in the repository (see [Limitations](#limitations)). It must be placed in `protocolosPDF/` and ingested via `build_rag_system.py`.

---

## Retrieval pipeline

### Ingestion

```
PDF → PyMuPDF (per-page text) → token chunking → BGE-M3 embeddings → ChromaDB
                                                        ↓
                                              metadata: source, page, chunk_id
```

Full build:

```powershell
python build_rag_system.py --variant bge
```

Incremental ingestion (skips unchanged documents via SHA-256 hash):

```powershell
python ingest_incremental.py
```

### Query flow

1. User submits a question via the web UI (`web_app.py`) or CLI (`chat.py`).
2. The query is classified (out-of-domain, comparison, multi-document, etc.).
3. Entities are extracted and the query is normalized (synonym/acronym expansion).
4. A BGE-M3 embedding is generated for the query.
5. Semantic search runs in ChromaDB; BM25 runs over the full chunk corpus.
6. Results are fused via weighted linear combination.
7. Top candidates are reranked by the cross-encoder.
8. Context is assembled and the factual gate checks for literal evidence.
9. The LLM (Ollama) generates an answer grounded in the context.
10. The answer is postprocessed: citations are extracted, sources are deduplicated, forbidden phrases are checked.

---

## Evaluation

The system is evaluated on a curated benchmark of **75 cybersecurity questions** with page-level ground truth. The benchmark covers five categories:

| Category | Count | Description |
|---|---|---|
| `simple` | 30 | Single-document factual questions |
| `multi_document` | 11 | Require combining 2–3 sources |
| `complex` | 12 | Long questions requiring synthesis |
| `no_answer` | 13 | Answer is not in the corpus (hallucination check) |
| `ambiguous` | 9 | Vague or double-interpretation questions |

### Methodology

The evaluation separates **retrieval** from **generation** so failures can be diagnosed independently:

- **Retrieval metrics** (deterministic, no LLM): Recall@K, MRR, Document Hit Rate, Page Hit Rate, Precision@K — computed from the ranked list of retrieved sources against the ground-truth `(source, page)` annotations.
- **End-to-end approval**: A question passes if retrieval finds the correct source, the answer contains expected keywords, no forbidden phrases appear, and citation markers match retrieved sources. For `no_answer` questions, the system must decline (detected via decline-phrase matching).
- **Factual gate**: A deterministic pattern-matching gate blocks answers to factual questions (CVEs, prices, RFCs, temperatures, passwords) when no literal evidence is found in the context. This is not an LLM judgment — it is a regex-based check.

The full benchmark requires the ingested corpus, local models, and a running Ollama instance, so it is **not part of CI**. It is documented in `tests/eval/` and the historical reports are committed under `tests/eval/reports/`.

### Results

The following metrics are from the evaluation report dated 2026-07-18 (`tests/eval/reports/report_20260718_054522_corrected.md`). These are historical results — the benchmark questions, ground truth, and reports are treated as immutable artifacts (see [Engineering Decisions](#engineering-decisions)).

**Retrieval metrics** (deterministic, computed from ranked sources):

| Metric | Value |
|---|---|
| Document Hit Rate (hit@K) | 82.5% |
| Page Hit Rate (hit@K, ±2 tolerance) | 77.8% |
| Recall@1 | 0.349 |
| Recall@3 | 0.619 |
| Recall@5 | 0.746 |
| MRR | 0.514 |
| Precision@K | 0.322 |

**End-to-end results** (retrieval + generation + anti-hallucination):

| Metric | Value |
|---|---|
| Questions evaluated | 75 |
| End-to-end approval | 94.7% (71/75) |
| Answerable approval rate | 96.5% (55/57) |
| No-answer approval rate (correct decline) | 69.2% (9/13) |
| Real hallucinations | 0 (1 blocked by factual gate) |

**Latency** (average per query):

| Stage | Avg time | % of total |
|---|---|---|
| LLM generation | 41,915 ms | 98.3% |
| Reranking | 3,491 ms | 8.2% |
| BM25 search | 1,145 ms | 2.7% |
| Query embedding | 744 ms | 1.7% |
| Semantic search | 9 ms | 0.0% |
| Fusion + ranking | 50 ms | 0.1% |
| **Total** | **42,641 ms** | |

**Notes on the end-to-end number**: The 94.7% end-to-end approval includes manual review of cases where the automatic harness marked a question as failed but the LLM produced a correct answer (e.g. the retriever found sufficient context from alternative sources, or the decline-phrase matcher didn't recognize a valid decline phrasing). The retrieval metrics (82.5% doc hit, 0.746 Recall@5) are fully automatic and were not adjusted. See the full report for the per-question breakdown.

---

## Engineering Decisions

- **Full benchmark execution is kept outside CI** because it requires multi-gigabyte local models (BGE-M3, BGE-reranker-v2-m3), an ingested ChromaDB corpus, and a running Ollama instance. CI uses deterministic synthetic data to provide fast feedback on the retrieval and fusion logic instead.
- **CI uses deterministic synthetic data** so that tests are reproducible on any machine without GPU, models, or external services.
- **The current repository layout is intentionally preserved** to minimize migration risk around the legacy retrieval orchestrator (`rag_hybrid.py`). A full `src/`-layout migration is planned as future work.
- **The quickstart intentionally avoids model dependencies** so the repository can be evaluated on a clean machine in under five minutes (see [Quickstart](#quickstart)).
- **Historical evaluation results are immutable.** The benchmark questions, ground-truth annotations, evaluation logic, and committed reports under `tests/eval/reports/` are not modified to improve reported numbers. Any methodology change is versioned and reported as a new evaluation.

---

## Repository structure

```
SistemaRAGHybrid/
├── rag_hybrid.py              # Main orchestrator (HybridRAG)
├── retrieval_engine.py        # Hybrid search + fusion + reranking
├── context_builder.py         # Context selection and prompt assembly
├── answer_postprocessor.py    # Response cleanup and citation extraction
├── query_classifier.py        # Intent classification
├── equivalences_manager.py    # Synonym/acronym expansion
├── ollama_manager.py          # Ollama lifecycle management
├── doc_cards.py               # Document role cards
├── memory_system.py           # SQLite knowledge/conversation memory
├── conceptual_map.py          # Conceptual shortcuts
├── learning_queue.py          # Deferred self-learning
├── web_app.py                 # Flask web interface
├── chat.py                    # CLI interface
├── build_rag_system.py        # Full corpus ingestion
├── ingest_incremental.py      # Incremental ingestion (hash-based)
├── query_rag.py               # Semantic-only query mode
├── config.yaml                # Central configuration
├── requirements.txt           # Python dependencies
├── src/
│   ├── pdf_extractor.py       # PyMuPDF per-page extraction
│   ├── chunker.py             # Token/semantic chunking
│   ├── embedder.py            # BGE-M3 embedding generation + cache
│   ├── vector_store.py        # ChromaDB persistence
│   ├── hash_registry.py       # Document hash registry
│   ├── rag/
│   │   ├── factual_gate.py    # Deterministic factual evidence gate
│   │   └── entity_extractor.py # NER and doc-reference extraction
│   └── utils/
│       ├── config_loader.py   # Centralized YAML config loading
│       ├── console.py         # Console utilities
│       └── device_utils.py    # CUDA/CPU device detection
├── tests/
│   ├── unit/                  # Unit tests (no models required)
│   ├── eval/                  # 75-question benchmark + reports
│   └── reports/               # Historical evaluation reports
├── examples/
│   └── quickstart/            # 5-minute demo (no models required)
├── scripts/                   # Diagnostics and utilities
├── tools/                     # Benchmarking tools
└── docs/                      # Additional documentation
```

---

## Installation

### Prerequisites

- Python 3.10+ (3.12 recommended)
- Ollama installed and available at `http://localhost:11434` (for the full pipeline)
- GPU with CUDA recommended (CPU works but ingestion and queries are slower)

### Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

If PowerShell blocks script execution, use the interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

### Local models (full pipeline only)

The full pipeline requires local models that are not downloaded by `pip`:

- **Embeddings**: `models/BAAI-bge-m3`
- **Reranker**: `models/BAAI-bge-reranker-v2-m3`
- **LLM**: any model available in Ollama (configured in `config.yaml`)

```powershell
ollama pull <model-name>
```

The quickstart and test suite do **not** require these models.

---

## Running the pipeline

### Ingest the corpus

1. Place PDFs in `protocolosPDF/`.
2. Ensure local models are present.
3. Run the build:

```powershell
python build_rag_system.py --variant bge
```

### Query

**Web UI** (Flask, default `http://localhost:5000`):

```powershell
python web_app.py
```

**CLI**:

```powershell
python chat.py
```

**Semantic-only query** (no LLM, shows retrieved chunks):

```powershell
python query_rag.py
```

---

## Running tests

```powershell
pytest
```

The test suite (257 tests, ~4s) runs without local models, ChromaDB, or Ollama. It covers:

- **Unit tests** (`tests/unit/`): chunking, entity extraction, query classification, fusion math, metadata contracts, config loading, embedder cache, postprocessing, prompt building.
- **Integration tests** (`tests/integration/`): BM25 indexing and hybrid retrieval (BM25 + mock semantic fusion) on a synthetic 5-document cybersecurity corpus.
- **Evaluation metric tests** (`tests/evaluation/`): deterministic verification of Recall@K, MRR, Doc/Page hit rate, precision@K, keyword scoring, hallucination detection, and citation fidelity against hand-computed expected values.

Tests that require local models or Ollama are marked `@pytest.mark.requires_models` / `@pytest.mark.requires_ollama` and are skipped by default.

### Full benchmark (manual, requires models + corpus)

```powershell
python tests/eval/run_cybersec_eval.py
```

See `tests/eval/README.md` for details.

### Continuous Integration

GitHub Actions runs the test suite and quickstart demo on every push and pull request to `main`, on Python 3.11 and 3.12 (`ubuntu-latest`). No models, GPU, or Ollama are required — only the deterministic unit, integration, and evaluation-metric tests plus the BM25 quickstart demo.

See `.github/workflows/tests.yml` for the workflow definition.

---

## Quickstart

A 5-minute demo that runs on a clean machine with no models, no GPU, and no Ollama:

```powershell
pip install -e ".[dev]"
python examples/quickstart/run_demo.py
```

This loads a 5-document cybersecurity corpus (~12 KB total), chunks it with `TextChunker` (19 chunks), builds a BM25 index, and runs 5 example queries. The entire demo completes in under 5 ms. It demonstrates the deterministic retrieval and chunking logic without embedding or LLM dependencies. See `examples/quickstart/README.md` for details.

---

## Docker

The Docker image runs the quickstart demo (TextChunker + BM25, no models required):

```bash
docker compose up
```

To run the test suite inside the container instead:

```bash
docker compose run hybrid-rag pytest -ra
```

**What the image includes**: Python 3.12, the project installed in editable mode with dev dependencies, the quickstart corpus, and the test suite.

**What the image does NOT include**: the BGE-M3 embedding model, the BGE-reranker-v2-m3 model, ChromaDB corpus data, or Ollama. The full pipeline requires these local resources and is not containerized — see [Limitations](#limitations) for details.

---

## Limitations

- **Full pipeline requires local resources**: the complete system needs the BGE-M3 embedding model (~2 GB), the BGE-reranker-v2-m3 model, a running Ollama instance, and the ingested ChromaDB corpus. These are not bundled in the repository.
- **Corpus is not included**: the ~861 PDFs must be provided separately and ingested via `build_rag_system.py`.
- **GPU recommended**: ingestion and embedding generation are significantly slower on CPU.
- **Windows-oriented**: the project was developed on Windows 10+ with PowerShell. It should work on Linux/macOS but has not been formally tested there.
- **Latency is dominated by the LLM**: average query time is ~42 seconds, of which 98.3% is LLM generation. This is a function of the local model size, not the retrieval pipeline.
- **No-answer detection is imperfect**: the system correctly declines 69.2% of unanswerable questions. The remaining failures involve decline phrasings not recognized by the harness or forbidden terms mentioned in the context of declining.
- **Repository layout is transitional**: the project has root-level scripts (e.g., `rag_hybrid.py`, `query_classifier.py`) and a `src/` directory with library modules. The `pyproject.toml` editable install makes `src/` modules importable as top-level modules (`import chunker`, `from rag.entity_extractor import ...`). Root-level modules remain accessible in tests via `conftest.py`. A full `src/`-layout migration that consolidates all modules into a single package hierarchy is planned as Future Work.

---

## Future work

- **Full `src/`-layout migration**: move root modules into a proper `src/hybrid_rag/` package and eliminate `sys.path` hacks.
- **Split `rag_hybrid.py`**: the main orchestrator is a single 297 KB file; decomposing it into focused components is a priority for maintainability.
- **Agentic RAG evolution**: this repository represents the Hybrid RAG stage of the project. The system is currently being evolved toward an Agentic RAG architecture with separated knowledge construction and runtime consumption.
- **Latency reduction**: target <30 seconds per query by optimizing LLM inference (quantization, smaller models, or batching).
- **Improved no-answer detection**: expand decline-phrase recognition and allow forbidden terms in decline context.
- **Cross-platform testing**: formal Linux/macOS CI support.
