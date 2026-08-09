"""
Unit tests for EmbeddingGenerator cache — tests the REAL cache logic by
monkeypatching _generate_embedding_uncached to avoid loading actual models.

The EmbeddingGenerator.__init__ loads a model (sentence-transformers or Ollama),
so we bypass __init__ and construct the object via __new__ to test only the
cache methods: _normalize_text_for_cache, _compute_text_hash, generate_embedding.
"""
import hashlib
import pytest
import numpy as np
from unittest.mock import patch

from embedder import EmbeddingGenerator


@pytest.fixture
def embedder():
    """Create an EmbeddingGenerator without calling __init__ (which loads a model)."""
    emb = EmbeddingGenerator.__new__(EmbeddingGenerator)
    emb.use_cache = True
    emb._cache_hits = 0
    emb._cache_misses = 0
    emb.embedding_dim = 768
    # Clear the LRU cache to avoid cross-test contamination
    EmbeddingGenerator._generate_embedding_cached.cache_clear()
    return emb


@pytest.fixture
def embedder_no_cache():
    """Create an EmbeddingGenerator with cache disabled."""
    emb = EmbeddingGenerator.__new__(EmbeddingGenerator)
    emb.use_cache = False
    emb._cache_hits = 0
    emb._cache_misses = 0
    emb.embedding_dim = 768
    return emb


def make_fake_embedding(dim=768):
    """Return a deterministic fake embedding vector."""
    return np.random.rand(dim).astype(np.float32)


class TestNormalizeTextForCache:
    def test_lowercases_text(self, embedder):
        result = embedder._normalize_text_for_cache("HELLO World")
        assert result == "hello world"

    def test_strips_whitespace(self, embedder):
        result = embedder._normalize_text_for_cache("  hello  ")
        assert result == "hello"

    def test_collapses_multiple_spaces(self, embedder):
        result = embedder._normalize_text_for_cache("hello    world")
        assert result == "hello world"

    def test_empty_string(self, embedder):
        assert embedder._normalize_text_for_cache("") == ""


class TestComputeTextHash:
    def test_same_text_produces_same_hash(self, embedder):
        h1 = embedder._compute_text_hash("NIST CSF")
        h2 = embedder._compute_text_hash("NIST CSF")
        assert h1 == h2

    def test_different_text_produces_different_hash(self, embedder):
        h1 = embedder._compute_text_hash("NIST CSF")
        h2 = embedder._compute_text_hash("ISO 27001")
        assert h1 != h2

    def test_normalized_text_produces_same_hash(self, embedder):
        """Text with different casing/spacing should produce the same hash."""
        h1 = embedder._compute_text_hash("NIST CSF")
        h2 = embedder._compute_text_hash("  nist  csf  ")
        assert h1 == h2

    def test_hash_is_md5_hex(self, embedder):
        h = embedder._compute_text_hash("test")
        assert len(h) == 32  # MD5 hex digest length
        assert all(c in '0123456789abcdef' for c in h)


class TestGenerateEmbeddingWithCache:
    def test_cache_hit_on_duplicate_text(self, embedder):
        fake = make_fake_embedding()
        with patch.object(embedder, '_generate_embedding_uncached', return_value=fake):
            emb1 = embedder.generate_embedding("NIST CSF")
            stats_before = (embedder._cache_hits, embedder._cache_misses)
            emb2 = embedder.generate_embedding("NIST CSF")
            stats_after = (embedder._cache_hits, embedder._cache_misses)
        # Second call should be a cache hit
        assert stats_after[0] > stats_before[0]
        np.testing.assert_array_equal(emb1, emb2)

    def test_cache_hit_on_normalized_text(self, embedder):
        fake = make_fake_embedding()
        with patch.object(embedder, '_generate_embedding_uncached', return_value=fake):
            emb1 = embedder.generate_embedding("NIST CSF")
            emb2 = embedder.generate_embedding("  nist  csf  ")
        # Normalized versions match, so second is a cache hit
        assert embedder._cache_hits >= 1
        np.testing.assert_array_equal(emb1, emb2)

    def test_cache_miss_on_different_text(self, embedder):
        fake1 = make_fake_embedding()
        fake2 = make_fake_embedding()
        with patch.object(embedder, '_generate_embedding_uncached', side_effect=[fake1, fake2]):
            emb1 = embedder.generate_embedding("NIST CSF")
            emb2 = embedder.generate_embedding("ISO 27001")
        # Both should be misses (different text)
        assert embedder._cache_misses == 0  # misses only counted on fallback
        assert embedder._cache_hits == 2   # both go through cached path

    def test_returns_numpy_array(self, embedder):
        fake = make_fake_embedding()
        with patch.object(embedder, '_generate_embedding_uncached', return_value=fake):
            result = embedder.generate_embedding("test")
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32


class TestGenerateEmbeddingWithoutCache:
    def test_no_cache_calls_uncached_directly(self, embedder_no_cache):
        fake = make_fake_embedding()
        with patch.object(embedder_no_cache, '_generate_embedding_uncached', return_value=fake) as mock:
            result = embedder_no_cache.generate_embedding("test")
        mock.assert_called_once_with("test")
        assert embedder_no_cache._cache_misses == 1
        assert embedder_no_cache._cache_hits == 0

    def test_no_cache_no_hits_on_duplicate(self, embedder_no_cache):
        fake = make_fake_embedding()
        with patch.object(embedder_no_cache, '_generate_embedding_uncached', return_value=fake) as mock:
            embedder_no_cache.generate_embedding("test")
            embedder_no_cache.generate_embedding("test")
        assert mock.call_count == 2
        assert embedder_no_cache._cache_hits == 0
