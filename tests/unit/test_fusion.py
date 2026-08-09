"""
Unit tests for hybrid fusion math — derived from retrieval_engine.hybrid_search.

The fusion logic in retrieval_engine.py computes:
    hybrid_score = semantic_weight * semantic_score + keyword_weight * keyword_score
where keyword_score is normalized by max BM25 score across candidates.

These tests verify the fusion math in isolation with synthetic scores,
without needing ChromaDB, embeddings, or the full RetrievalEngine.
"""
import pytest
import numpy as np


def fuse_scores(semantic_scores: dict, bm25_scores: dict, semantic_weight: float = 0.6) -> list:
    """
    Replicates the fusion logic from retrieval_engine.hybrid_search (lines 159-186).

    Args:
        semantic_scores: {index: semantic_score} (1 - cosine_distance)
        bm25_scores: {index: bm25_score}
        semantic_weight: weight for semantic score (keyword_weight = 1 - semantic_weight)

    Returns:
        List of (index, hybrid_score) sorted descending.
    """
    keyword_weight = 1 - semantic_weight
    cand_idx = set(semantic_scores.keys()) | set(bm25_scores.keys())
    max_bm25 = max((bm25_scores.get(i, 0.0) for i in cand_idx), default=1.0)
    if max_bm25 <= 0:
        max_bm25 = 1.0
    results = []
    for i in cand_idx:
        semantic_score = float(semantic_scores.get(i, 0.0))
        keyword_score = float(bm25_scores.get(i, 0.0)) / max_bm25
        hybrid_score = semantic_weight * semantic_score + keyword_weight * keyword_score
        results.append((i, hybrid_score))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


class TestFusionMath:
    def test_pure_semantic_score_dominates_with_high_weight(self):
        sem = {0: 0.9, 1: 0.1}
        bm25 = {0: 0.0, 1: 10.0}
        results = fuse_scores(sem, bm25, semantic_weight=0.99)
        # With 0.99 semantic weight, the high-semantic item should win
        assert results[0][0] == 0

    def test_pure_bm25_score_dominates_with_low_weight(self):
        sem = {0: 0.9, 1: 0.1}
        bm25 = {0: 0.0, 1: 10.0}
        results = fuse_scores(sem, bm25, semantic_weight=0.01)
        # With 0.01 semantic weight, the high-BM25 item should win
        assert results[0][0] == 1

    def test_equal_scores_produce_equal_hybrid(self):
        sem = {0: 0.5, 1: 0.5}
        bm25 = {0: 5.0, 1: 5.0}
        results = fuse_scores(sem, bm25, semantic_weight=0.5)
        assert results[0][1] == pytest.approx(results[1][1])

    def test_bm25_normalization_by_max(self):
        sem = {0: 0.0, 1: 0.0}
        bm25 = {0: 5.0, 1: 10.0}
        results = fuse_scores(sem, bm25, semantic_weight=0.0)
        # With 0 semantic weight, keyword_score = bm25/max_bm25
        # Item 1 has bm25=10 (max), so keyword_score=1.0
        # Item 0 has bm25=5, so keyword_score=0.5
        scores = {idx: score for idx, score in results}
        assert scores[1] == pytest.approx(1.0)
        assert scores[0] == pytest.approx(0.5)

    def test_candidate_union_of_semantic_and_bm25(self):
        sem = {0: 0.8, 2: 0.3}
        bm25 = {1: 5.0, 2: 2.0}
        results = fuse_scores(sem, bm25, semantic_weight=0.5)
        indices = {idx for idx, _ in results}
        assert indices == {0, 1, 2}

    def test_missing_semantic_score_treated_as_zero(self):
        sem = {0: 0.8}
        bm25 = {0: 5.0, 1: 10.0}
        results = fuse_scores(sem, bm25, semantic_weight=0.5)
        scores = {idx: score for idx, score in results}
        # Item 1 has no semantic score -> 0
        # Its hybrid = 0.5*0 + 0.5*(10/10) = 0.5
        assert scores[1] == pytest.approx(0.5)

    def test_missing_bm25_score_treated_as_zero(self):
        sem = {0: 0.8, 1: 0.4}
        bm25 = {0: 5.0}
        results = fuse_scores(sem, bm25, semantic_weight=0.5)
        scores = {idx: score for idx, score in results}
        # Item 1 has no BM25 score -> 0
        # Its hybrid = 0.5*0.4 + 0.5*0 = 0.2
        assert scores[1] == pytest.approx(0.2)

    def test_all_zero_bm25_does_not_divide_by_zero(self):
        sem = {0: 0.8}
        bm25 = {0: 0.0}
        # Should not raise
        results = fuse_scores(sem, bm25, semantic_weight=0.5)
        assert len(results) == 1

    def test_empty_inputs_produce_empty_results(self):
        results = fuse_scores({}, {}, semantic_weight=0.6)
        assert results == []

    def test_default_semantic_weight_is_0_6(self):
        """The config default is semantic_weight=0.6, keyword_weight=0.4."""
        sem = {0: 1.0}
        bm25 = {0: 10.0}
        results = fuse_scores(sem, bm25, semantic_weight=0.6)
        # hybrid = 0.6*1.0 + 0.4*(10/10) = 0.6 + 0.4 = 1.0
        assert results[0][1] == pytest.approx(1.0)

    def test_results_sorted_descending(self):
        sem = {0: 0.3, 1: 0.9, 2: 0.1}
        bm25 = {0: 3.0, 1: 1.0, 2: 9.0}
        results = fuse_scores(sem, bm25, semantic_weight=0.5)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_only_semantic_candidates(self):
        """Items only in semantic results (no BM25 score) still appear."""
        sem = {0: 0.9, 1: 0.5, 2: 0.3}
        bm25 = {}
        results = fuse_scores(sem, bm25, semantic_weight=0.6)
        assert len(results) == 3
        # With no BM25, max_bm25 defaults to 1.0, keyword_score = 0
        # hybrid = 0.6 * semantic_score
        scores = {idx: score for idx, score in results}
        assert scores[0] == pytest.approx(0.6 * 0.9)
