"""
Integration test: BM25 indexing and retrieval on a synthetic corpus.

This is the deterministic backbone of the integration test suite.
No embeddings, no ChromaDB, no models — just BM25 + tokenization.
"""
import pytest
from rank_bm25 import BM25Okapi


def tokenize(text):
    import re
    return [t for t in re.split(r'\W+', text.lower()) if t]


@pytest.fixture
def index(corpus):
    tokenized = [tokenize(doc["text"]) for doc in corpus]
    return BM25Okapi(tokenized)


class TestBM25Index:
    def test_index_built_from_corpus(self, index, corpus):
        # BM25Okapi stores corpus_size
        assert index.corpus_size == len(corpus)

    def test_query_returns_scores_for_all_docs(self, index, corpus):
        scores = index.get_scores(tokenize("firewall network security"))
        assert len(scores) == len(corpus)

    def test_keyword_query_ranks_relevant_doc_first(self, index, corpus):
        query = "firewall"
        scores = index.get_scores(tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        top_doc_idx = ranked[0][0]
        assert "firewall" in corpus[top_doc_idx]["text"].lower()

    def test_multi_word_query_ranks_relevant_doc(self, index, corpus):
        query = "NIST cybersecurity framework functions"
        scores = index.get_scores(tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        top_doc_idx = ranked[0][0]
        assert "NIST" in corpus[top_doc_idx]["text"]

    def test_zero_score_for_non_matching_query(self, index, corpus):
        query = "minecraft recipe cooking"
        scores = index.get_scores(tokenize(query))
        # All scores should be 0 or very low for non-matching query
        assert max(scores) <= 0.01

    def test_top_k_retrieval(self, index, corpus):
        query = "security controls domains"
        scores = index.get_scores(tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        top_3 = ranked[:3]
        assert len(top_3) == 3
        # The top result should mention "controls" or "domains"
        top_text = corpus[top_3[0][0]]["text"].lower()
        assert "control" in top_text or "domain" in top_text

    def test_deterministic_results(self, index, corpus):
        """Same query should always produce the same ranking."""
        query = "zero trust verify"
        scores1 = index.get_scores(tokenize(query))
        scores2 = index.get_scores(tokenize(query))
        assert list(scores1) == list(scores2)

    def test_empty_query_returns_zero_scores(self, index, corpus):
        scores = index.get_scores(tokenize(""))
        assert all(s == 0 for s in scores)

    def test_ciissp_query_finds_cissp_doc(self, index, corpus):
        query = "CISSP domains certification"
        scores = index.get_scores(tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        top_doc_idx = ranked[0][0]
        assert "CISSP" in corpus[top_doc_idx]["text"]
