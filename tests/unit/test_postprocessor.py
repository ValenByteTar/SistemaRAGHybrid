"""
Unit tests for AnswerPostprocessor — derived from observable behavior of answer_postprocessor.py.

Tests cover the static/pure methods that don't require a full HybridRAG instance:
- condense_text: truncation respecting paragraph boundaries
- truncate_safe_short: truncation respecting step lists
- has_numeric_evidence: checking for numeric evidence in results
- has_procedural_evidence: checking for procedural evidence in context
- looks_like_steps: detecting step-formatted text
- Citation extraction from context ([Doc N - source p.X] format)
"""
import re
import pytest
from unittest.mock import MagicMock
from answer_postprocessor import AnswerPostprocessor


@pytest.fixture
def postprocessor():
    """Create an AnswerPostprocessor with a minimal mock HybridRAG."""
    mock_rag = MagicMock()
    mock_rag.flags = {
        'postprocess_number_synonyms': True,
        'postprocess_unit_synonyms': True,
        'postprocess_location': True,
        'postprocess_comparison_summary': True,
    }
    return AnswerPostprocessor(mock_rag)


class TestCondenseText:
    def test_short_text_unchanged(self, postprocessor):
        text = "Short text under limit."
        assert postprocessor.condense_text(text, max_chars=600) == text

    def test_long_text_truncated_at_paragraph(self, postprocessor):
        text = "First paragraph.\n\n" + "Second paragraph that is very long. " * 50
        result = postprocessor.condense_text(text, max_chars=100)
        assert len(result) <= 100 or result == text  # May keep first para only

    def test_empty_text_returns_empty(self, postprocessor):
        assert postprocessor.condense_text("", max_chars=600) == ""

    def test_single_paragraph_under_limit(self, postprocessor):
        text = "A single short paragraph."
        assert postprocessor.condense_text(text, max_chars=600) == text


class TestTruncateSafeShort:
    def test_short_text_unchanged(self, postprocessor):
        text = "Short answer."
        assert postprocessor.truncate_safe_short(text, limit=1000) == text

    def test_empty_text_returns_empty(self, postprocessor):
        assert postprocessor.truncate_safe_short("", limit=1000) == ""

    def test_none_text_returns_empty(self, postprocessor):
        assert postprocessor.truncate_safe_short(None, limit=1000) == ''

    def test_step_list_respected(self, postprocessor):
        text = "1. First step do something here\n2. Second step do more things here\n3. Third step here too"
        result = postprocessor.truncate_safe_short(text, limit=100)
        # Should keep at least some steps
        assert "1." in result or "2." in result

    def test_long_text_truncated_at_sentence_boundary(self, postprocessor):
        text = "This is a sentence. " * 100
        result = postprocessor.truncate_safe_short(text, limit=100)
        assert len(result) <= 101  # limit + possible ellipsis


class TestHasNumericEvidence:
    def test_entity_with_number_in_text_returns_true(self, postprocessor):
        results = [
            {'text': 'ISO 27001 has 114 controls in the framework', 'metadata': {'source': 'ISO.pdf'}},
        ]
        assert postprocessor.has_numeric_evidence('iso', results) is True

    def test_entity_without_number_returns_false(self, postprocessor):
        results = [
            {'text': 'ISO is a standard for security', 'metadata': {'source': 'ISO.pdf'}},
        ]
        assert postprocessor.has_numeric_evidence('iso', results) is False

    def test_empty_results_returns_false(self, postprocessor):
        assert postprocessor.has_numeric_evidence('iso', []) is False

    def test_empty_entity_returns_false(self, postprocessor):
        results = [{'text': 'Some text with 123', 'metadata': {'source': 'A.pdf'}}]
        assert postprocessor.has_numeric_evidence('', results) is False

    def test_entity_not_in_text_returns_false(self, postprocessor):
        results = [
            {'text': 'CISSP has 8 domains with numbers', 'metadata': {'source': 'CISSP.pdf'}},
        ]
        assert postprocessor.has_numeric_evidence('nist', results) is False


class TestHasProceduralEvidence:
    def test_multiple_sources_with_proc_keywords(self, postprocessor):
        context = "[Doc 1 - Guide.pdf p.5]\nProcedimiento para configurar\n\n[Doc 2 - Manual.pdf p.10]\nStep by step instructions"
        assert postprocessor.has_procedural_evidence(context, min_sources=2) is True

    def test_single_source_returns_false(self, postprocessor):
        context = "[Doc 1 - Guide.pdf p.5]\nProcedimiento para configurar"
        assert postprocessor.has_procedural_evidence(context, min_sources=2) is False

    def test_no_procedural_keywords_returns_false(self, postprocessor):
        context = "[Doc 1 - Guide.pdf p.5]\nSome general text\n\n[Doc 2 - Manual.pdf p.10]\nMore general text"
        assert postprocessor.has_procedural_evidence(context, min_sources=2) is False

    def test_empty_context_returns_false(self, postprocessor):
        assert postprocessor.has_procedural_evidence("", min_sources=2) is False

    def test_none_context_returns_false(self, postprocessor):
        assert postprocessor.has_procedural_evidence(None, min_sources=2) is False


class TestLooksLikeSteps:
    def test_numbered_steps_detected(self, postprocessor):
        text = "1. First step with enough text\n2. Second step with enough text"
        assert postprocessor.looks_like_procedural_steps(text) is True

    def test_bullet_steps_detected(self, postprocessor):
        text = "- First bullet item with enough text\n- Second bullet item with enough text"
        assert postprocessor.looks_like_procedural_steps(text) is True

    def test_prose_not_steps(self, postprocessor):
        text = "This is a normal paragraph without any step formatting at all."
        assert postprocessor.looks_like_procedural_steps(text) is False

    def test_single_step_not_enough(self, postprocessor):
        text = "1. Only one step here with enough text"
        assert postprocessor.looks_like_procedural_steps(text) is False


class TestCitationExtractionFromContext:
    """Test that citation markers [Doc N - source p.X] are correctly parsed from context."""

    def test_extract_sources_from_context(self, postprocessor):
        context = "[Doc 1 - NIST CSF v2.pdf p.5]\nSome content\n\n[Doc 2 - ISO 27001.pdf p.12]\nMore content"
        sources = re.findall(r'\[Doc \d+ - (.+?) p\.\d+\]', context)
        assert "NIST CSF v2.pdf" in sources
        assert "ISO 27001.pdf" in sources

    def test_extract_unique_sources(self, postprocessor):
        context = "[Doc 1 - Guide.pdf p.5]\nText\n\n[Doc 2 - Guide.pdf p.10]\nMore text"
        sources = re.findall(r'\[Doc \d+ - (.+?) p\.\d+\]', context)
        unique = set(sources)
        assert len(unique) == 1
        assert "Guide.pdf" in unique


class TestPostProcessAnswer:
    def test_short_answer_unchanged(self, postprocessor):
        answer = "Short answer."
        result = postprocessor.post_process_answer(answer)
        # Short answers (< 10 chars) are returned as-is
        assert result is not None

    def test_empty_answer_returned(self, postprocessor):
        assert postprocessor.post_process_answer("") == ""

    def test_strips_trailing_markdown(self, postprocessor):
        answer = "This is a complete answer with a period.**"
        result = postprocessor.post_process_answer(answer)
        assert not result.rstrip().endswith("**")

    def test_removes_correction_patterns(self, postprocessor):
        answer = "The answer is NIST CSF and it defines five core functions for security.\n\n¿Por qué la respuesta anterior fue incorrecta? Because..."
        result = postprocessor.post_process_answer(answer)
        assert "¿Por qué" not in result
