# Hybrid RAG — Quickstart Demo Image
#
# Runs the deterministic quickstart demo (TextChunker + BM25) without
# embeddings, ChromaDB, Ollama, or GPU. This is NOT the full pipeline image.
#
# The full pipeline requires local models (BGE-M3, BGE-reranker-v2-m3) and a
# running Ollama instance on the host. Those are documented in the README and
# are intentionally not bundled here.

FROM python:3.12-slim

WORKDIR /app

# Install build dependencies for editable install
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy project metadata first for better layer caching
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY examples/ ./examples/
COPY tests/ ./tests/
COPY conftest.py ./

# Install the package (editable) with dev extras for pytest
RUN pip install --no-cache-dir -e ".[dev]"

# Default entrypoint: run the quickstart demo
ENTRYPOINT ["python", "examples/quickstart/run_demo.py"]
