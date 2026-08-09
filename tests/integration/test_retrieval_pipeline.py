"""
Integration test: retrieval pipeline with BM25 + fusion.

Tests the end-to-end retrieval flow:
  corpus → tokenize → BM25 index → query → fuse with mock semantic scores → rank

No ChromaDB, no embeddings model, no LLM. The mock embedder provides
deterministic hash-based vectors so the fusion logic can be exercised
without GPU or model downloads.
"""
import heapq
import pytest
import numpy as np


def tokenize(text):
    import re
    return [t for t in re.split(r'\W+', text.lower()) if t]


def hybrid_search(query, corpus, bm25_index, mock_embedder, top_k=5, semantic_weight=0.6):
    """
    Replicates the core fusion logic from retrieval_engine.hybrid_search
    using BM25 + mock semantic scores (no ChromaDB needed).
    """
    keyword_weight = 1 - semantic_weight

    # BM25 scores
    bm25_scores = bm25_index.get_scores(tokenize(query))

    # Mock semantic scores: cosine similarity between query and doc embeddings
    query_emb = mock_embedder.generate_embedding(query)
    sem_scores = []
    for doc in corpus:
        doc_emb = mock_embedder.generate_embedding(doc["text"])
        # Cosine similarity (embeddings are already normalized)
        sim = float(np.dot(query_emb, doc_emb))
        sem_scores.append(sim)

    # Fusion (same logic as retrieval_engine.py)
    max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
    results = []
    for i in range(len(corpus)):
        semantic_score = sem_scores[i]
        keyword_score = bm25_scores[i] / max_bm25
        hybrid_score = semantic_weight * semantic_score + keyword_weight * keyword_score
        results.append({
            'index': i,
            'text': corpus[i]['text'],
            'metadata': {'source': corpus[i]['source'], 'page': corpus[i]['page']},
            'hybrid_score': hybrid_score,
            'semantic_score': semantic_score,
            'keyword_score': keyword_score,
        })

    top = heapq.nlargest(top_k, results, key=lambda x: x['hybrid_score'])
    return top


class TestHybridSearch:
    def test_returns_top_k_results(self, corpus, bm25_index, mock_embedder):
        results = hybrid_search("firewall", corpus, bm25_index, mock_embedder, top_k=3)
        assert len(results) == 3

    def test_keyword_dominant_query_finds_relevant_doc(self, corpus, bm25_index, mock_embedder):
        """With low semantic weight, BM25 should dominate."""
        results = hybrid_search(
            "firewall network security device",
            corpus, bm25_index, mock_embedder,
            top_k=1, semantic_weight=0.01,
        )
        assert "firewall" in results[0]['text'].lower()

    def test_results_have_hybrid_score(self, corpus, bm25_index, mock_embedder):
        results = hybrid_search("NIST framework", corpus, bm25_index, mock_embedder, top_k=3)
        for r in results:
            assert 'hybrid_score' in r
            assert isinstance(r['hybrid_score'], float)

    def test_results_have_metadata(self, corpus, bm25_index, mock_embedder):
        results = hybrid_search("ISO 27001", corpus, bm25_index, mock_embedder, top_k=2)
        for r in results:
            assert 'metadata' in r
            assert 'source' in r['metadata']
            assert 'page' in r['metadata']

    def test_results_sorted_by_hybrid_score_descending(self, corpus, bm25_index, mock_embedder):
        results = hybrid_search("security", corpus, bm25_index, mock_embedder, top_k=5)
        scores = [r['hybrid_score'] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_deterministic_results(self, corpus, bm25_index, mock_embedder):
        """Same query should always produce the same ranking."""
        r1 = hybrid_search("NIST CSF", corpus, bm25_index, mock_embedder, top_k=3)
        r2 = hybrid_search("NIST CSF", corpus, bm25_index, mock_embedder, top_k=3)
        assert [r['index'] for r in r1] == [r['index'] for r in r2]

    def test_empty_query_returns_results(self, corpus, bm25_index, mock_embedder):
        """Empty query should still return results (all scores may be 0 but no crash)."""
        results = hybrid_search("", corpus, bm25_index, mock_embedder, top_k=3)
        assert len(results) == 3

    def test_semantic_and_keyword_scores_present(self, corpus, bm25_index, mock_embedder):
        results = hybrid_search("zero trust", corpus, bm25_index, mock_embedder, top_k=2)
        for r in results:
            assert 'semantic_score' in r
            assert 'keyword_score' in r

    def test_fusion_respects_weights(self, corpus, bm25_index, mock_embedder):
        """Verify hybrid_score = sw * semantic + kw * keyword."""
        results = hybrid_search("test", corpus, bm25_index, mock_embedder, top_k=1, semantic_weight=0.6)
        r = results[0]
        expected = 0.6 * r['semantic_score'] + 0.4 * r['keyword_score']
        assert r['hybrid_score'] == pytest.approx(expected, rel=1e-5)

    def test_zero_trust_query_finds_zero_trust_doc(self, corpus, bm25_index, mock_embedder):
        results = hybrid_search(
            "zero trust architecture verify",
            corpus, bm25_index, mock_embedder,
            top_k=1, semantic_weight=0.01,  # BM25 dominant
        )
        assert "Zero Trust" in results[0]['text'] or "zero trust" in results[0]['text'].lower()
