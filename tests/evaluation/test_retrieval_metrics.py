"""
Unit tests for retrieval evaluation metrics — verifies the deterministic
metric functions from tests/eval/run_cybersec_eval.py against hand-computed
expected values.

These tests prove the metric math is correct without needing the corpus,
models, or LLM. They test:
- validate_retrieval: hit_doc, hit_page, recall, MRR, precision_at_k
- validate_response_keywords: keyword_score, forbidden detection
- validate_hallucination: decline detection, hallucination check
- validate_citation_fidelity: citation verification
- extract_citation_sources: citation marker parsing
- normalize: accent-insensitive normalization
"""
import pytest
import sys
from pathlib import Path

# The eval harness lives in tests/eval/ — add it to path
EVAL_DIR = Path(__file__).parent.parent / "eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from run_cybersec_eval import (
    validate_retrieval,
    validate_response_keywords,
    validate_hallucination,
    validate_citation_fidelity,
    extract_citation_sources,
    normalize,
)


# ---------------------------------------------------------------------------
# validate_retrieval
# ---------------------------------------------------------------------------

class TestValidateRetrieval:
    def test_hit_doc_when_expected_source_in_results(self):
        sources_api = [
            {"name": "NIST CSF v2.pdf", "page": 5, "score": 0.9},
        ]
        result = validate_retrieval(sources_api, ["NIST CSF v2.pdf"], [5], tolerance=2)
        assert result["hit_doc"] is True

    def test_miss_doc_when_expected_source_not_in_results(self):
        sources_api = [
            {"name": "Other.pdf", "page": 1, "score": 0.9},
        ]
        result = validate_retrieval(sources_api, ["NIST CSF v2.pdf"], [5], tolerance=2)
        assert result["hit_doc"] is False

    def test_hit_page_when_page_within_tolerance(self):
        sources_api = [{"name": "NIST CSF v2.pdf", "page": 6, "score": 0.9}]
        result = validate_retrieval(sources_api, ["NIST CSF v2.pdf"], [5], tolerance=2)
        assert result["hit_page"] is True

    def test_miss_page_when_page_outside_tolerance(self):
        sources_api = [{"name": "NIST CSF v2.pdf", "page": 20, "score": 0.9}]
        result = validate_retrieval(sources_api, ["NIST CSF v2.pdf"], [5], tolerance=2)
        assert result["hit_page"] is False

    def test_recall_single_doc_found(self):
        sources_api = [{"name": "NIST CSF v2.pdf", "page": 5}]
        result = validate_retrieval(sources_api, ["NIST CSF v2.pdf"], [5], tolerance=2)
        assert result["recall"] == 1.0

    def test_recall_multi_doc_partial(self):
        sources_api = [
            {"name": "NIST CSF v2.pdf", "page": 5},
            {"name": "Other.pdf", "page": 1},
        ]
        result = validate_retrieval(sources_api, ["NIST CSF v2.pdf", "ISO 27001.pdf"], [5, 12], tolerance=2)
        assert result["recall"] == 0.5

    def test_mrr_rank_1(self):
        sources_api = [{"name": "NIST CSF v2.pdf", "page": 5}]
        result = validate_retrieval(sources_api, ["NIST CSF v2.pdf"], [5], tolerance=2)
        assert result["mrr"] == 1.0
        assert result["first_relevant_rank"] == 1

    def test_mrr_rank_3(self):
        sources_api = [
            {"name": "Other1.pdf", "page": 1},
            {"name": "Other2.pdf", "page": 2},
            {"name": "NIST CSF v2.pdf", "page": 5},
        ]
        result = validate_retrieval(sources_api, ["NIST CSF v2.pdf"], [5], tolerance=2)
        assert result["mrr"] == pytest.approx(1/3, abs=0.001)
        assert result["first_relevant_rank"] == 3

    def test_mrr_zero_when_not_found(self):
        sources_api = [{"name": "Other.pdf", "page": 1}]
        result = validate_retrieval(sources_api, ["NIST CSF v2.pdf"], [5], tolerance=2)
        assert result["mrr"] == 0.0
        assert result["first_relevant_rank"] is None

    def test_precision_at_k(self):
        sources_api = [
            {"name": "NIST CSF v2.pdf", "page": 5},
            {"name": "ISO 27001.pdf", "page": 12},
            {"name": "Irrelevant.pdf", "page": 1},
        ]
        expected = ["NIST CSF v2.pdf", "ISO 27001.pdf"]
        result = validate_retrieval(sources_api, expected, [5, 12], tolerance=2)
        # 2 relevant out of 3 retrieved
        assert result["precision_at_k"] == pytest.approx(2/3, abs=0.001)

    def test_empty_expected_sources_returns_skipped(self):
        result = validate_retrieval([], [], [], tolerance=2)
        assert result["skipped"] is True
        assert result["hit_doc"] is None

    def test_page_list_tolerance(self):
        """expected_pages can be a list of acceptable pages."""
        sources_api = [{"name": "CISSP Guide.pdf", "page": 32}]
        result = validate_retrieval(sources_api, ["CISSP Guide.pdf"], [[30, 36]], tolerance=2)
        assert result["hit_page"] is True

    def test_partial_source_match(self):
        """Source matching is bidirectional partial (substring)."""
        sources_api = [{"name": "NIST CSF v2 Final 2024.pdf", "page": 5}]
        result = validate_retrieval(sources_api, ["NIST CSF v2"], [5], tolerance=2)
        assert result["hit_doc"] is True


# ---------------------------------------------------------------------------
# validate_response_keywords
# ---------------------------------------------------------------------------

class TestValidateResponseKeywords:
    def test_all_keywords_present(self):
        result = validate_response_keywords("NIST framework cybersecurity", ["nist", "framework"], [])
        assert result["keyword_score"] == 1.0
        assert result["keywords_pass"] is True

    def test_partial_keywords(self):
        result = validate_response_keywords("NIST is a framework", ["nist", "framework", "cybersecurity"], [])
        assert result["keyword_score"] == pytest.approx(2/3, abs=0.001)

    def test_no_keywords_present(self):
        result = validate_response_keywords("Something else", ["nist", "framework"], [])
        assert result["keyword_score"] == 0.0

    def test_forbidden_phrase_detected(self):
        result = validate_response_keywords("lo siento, no hay informacion", [], ["lo siento"])
        assert result["forbidden_pass"] is False
        assert "lo siento" in result["found_forbidden"]

    def test_no_forbidden_passes(self):
        result = validate_response_keywords("NIST CSF defines functions", ["nist"], [])
        assert result["forbidden_pass"] is True

    def test_empty_keywords_returns_perfect_score(self):
        result = validate_response_keywords("any answer", [], [])
        assert result["keyword_score"] == 1.0


# ---------------------------------------------------------------------------
# validate_hallucination
# ---------------------------------------------------------------------------

class TestValidateHallucination:
    def test_answerable_not_applicable(self):
        result = validate_hallucination("Some answer", is_answerable=True, forbidden=[])
        assert result["applicable"] is False

    def test_declined_correctly_passes(self):
        result = validate_hallucination("No hay evidencia suficiente", is_answerable=False, forbidden=["salario"])
        assert result["declined"] is True
        assert result["hallucinated"] is False
        assert result["pass"] is True

    def test_hallucinated_fails(self):
        result = validate_hallucination("El salario es $5000", is_answerable=False, forbidden=["salario"])
        assert result["declined"] is False
        assert result["hallucinated"] is True
        assert result["pass"] is False

    def test_no_decline_and_no_forbidden_fails(self):
        result = validate_hallucination("Some random answer", is_answerable=False, forbidden=[])
        assert result["declined"] is False
        assert result["pass"] is False


# ---------------------------------------------------------------------------
# validate_citation_fidelity
# ---------------------------------------------------------------------------

class TestValidateCitationFidelity:
    def test_all_citations_verified(self):
        answer = "See [Doc 1 - NIST CSF v2.pdf p.5] and [Doc 2 - ISO 27001.pdf p.12]"
        sources = [
            {"name": "NIST CSF v2.pdf", "page": 5},
            {"name": "ISO 27001.pdf", "page": 12},
        ]
        result = validate_citation_fidelity(answer, sources)
        assert result["score"] == 1.0
        assert len(result["verified"]) == 2
        assert len(result["unverified"]) == 0

    def test_unverified_citation_detected(self):
        answer = "See [Doc 1 - Unknown.pdf p.5]"
        sources = [{"name": "NIST CSF v2.pdf", "page": 5}]
        result = validate_citation_fidelity(answer, sources)
        assert result["score"] == 0.0
        assert len(result["unverified"]) == 1

    def test_no_citations_returns_none_score(self):
        result = validate_citation_fidelity("Answer without citations", [])
        assert result["score"] is None
        assert result["cited"] == []

    def test_partial_verification(self):
        answer = "See [Doc 1 - NIST CSF v2.pdf p.5] and [Doc 2 - Fake.pdf p.1]"
        sources = [{"name": "NIST CSF v2.pdf", "page": 5}]
        result = validate_citation_fidelity(answer, sources)
        assert result["score"] == 0.5
        assert len(result["verified"]) == 1
        assert len(result["unverified"]) == 1


# ---------------------------------------------------------------------------
# extract_citation_sources
# ---------------------------------------------------------------------------

class TestExtractCitationSources:
    def test_single_citation(self):
        sources = extract_citation_sources("Text [Doc 1 - Guide.pdf p.5] more text")
        assert sources == ["Guide.pdf"]

    def test_multiple_citations(self):
        sources = extract_citation_sources("[Doc 1 - A.pdf p.1] and [Doc 2 - B.pdf p.2]")
        assert sources == ["A.pdf", "B.pdf"]

    def test_no_citations(self):
        assert extract_citation_sources("No citations here") == []

    def test_case_insensitive(self):
        sources = extract_citation_sources("[doc 3 - test.pdf p.10]")
        assert sources == ["test.pdf"]


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_strips_accents(self):
        assert normalize("NIST Ciberseguridad") == "nist ciberseguridad"

    def test_lowercases(self):
        assert normalize("HELLO") == "hello"

    def test_combined(self):
        assert normalize("Ciberseguridád ÍSO") == "ciberseguridad iso"

    def test_empty_string(self):
        assert normalize("") == ""
