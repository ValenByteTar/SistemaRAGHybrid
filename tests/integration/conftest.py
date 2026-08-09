"""
Test fixtures for integration tests.

Provides a tiny synthetic corpus, BM25 index, and mock embedder for
deterministic integration testing without GPU, models, or Ollama.
"""
import hashlib
import pytest
import numpy as np
from rank_bm25 import BM25Okapi


# --- Synthetic cybersecurity corpus ---

SYNTHETIC_CORPUS = [
    {
        "id": "doc_0",
        "text": "The NIST Cybersecurity Framework defines five core functions: Identify, Protect, Detect, Respond, and Recover. It provides a structured approach to managing cybersecurity risk.",
        "source": "NIST_CSF_Overview.txt",
        "page": 1,
    },
    {
        "id": "doc_1",
        "text": "ISO 27001 is an international standard for information security management systems. It contains 114 controls organized into 14 domains including access control and cryptography.",
        "source": "ISO_27001_Guide.txt",
        "page": 5,
    },
    {
        "id": "doc_2",
        "text": "A firewall is a network security device that monitors and filters incoming and outgoing network traffic based on an organization's security policies. Firewalls can be hardware or software.",
        "source": "Firewall_Basics.txt",
        "page": 3,
    },
    {
        "id": "doc_3",
        "text": "The CISSP certification covers eight domains: Security and Risk Management, Asset Security, Security Architecture and Engineering, Communication and Network Security, Identity and Access Management, Security Assessment and Testing, Security Operations, and Software Development Security.",
        "source": "CISSP_Study_Guide.txt",
        "page": 30,
    },
    {
        "id": "doc_4",
        "text": "Zero Trust Architecture assumes that threats exist both inside and outside the network. The core principle is never trust, always verify. Every access request must be authenticated and authorized.",
        "source": "Zero_Trust_Model.txt",
        "page": 2,
    },
]


def tokenize(text):
    """Simple tokenizer for BM25: lowercase, split on non-alphanumeric."""
    import re
    return [t for t in re.split(r'\W+', text.lower()) if t]


@pytest.fixture
def corpus():
    """Return the synthetic corpus."""
    return SYNTHETIC_CORPUS


@pytest.fixture
def bm25_index(corpus):
    """Build a BM25 index from the synthetic corpus."""
    tokenized = [tokenize(doc["text"]) for doc in corpus]
    return BM25Okapi(tokenized)


@pytest.fixture
def mock_embedder():
    """Return a mock embedder that produces deterministic hash-based vectors."""
    class MockEmbedder:
        def __init__(self):
            self.embedding_dim = 128

        def generate_embedding(self, text):
            """Generate a deterministic embedding from text hash."""
            h = hashlib.md5(text.lower().encode('utf-8')).digest()
            # Repeat hash bytes to fill the embedding dimension
            repeats = (self.embedding_dim + len(h) - 1) // len(h)
            raw = (h * repeats)[:self.embedding_dim]
            vec = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
            # Normalize to unit vector
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            return vec

    return MockEmbedder()
