"""
Unit tests for ContextBuilder prompt assembly — derived from context_builder.py.

Tests cover:
- build_context_from_results: basic context concatenation
- build_structured_context: categorized context with [Doc N] numbering
- build_focus_prompt: FOCUS instructions for the LLM
- collect_snippets_for_llm_scoring: snippet preparation
"""
import pytest
from unittest.mock import MagicMock
from context_builder import ContextBuilder


@pytest.fixture
def builder():
    """Create a ContextBuilder with a minimal mock HybridRAG."""
    mock_rag = MagicMock()
    mock_rag.ollama_model = "test-model"
    mock_rag.num_gpu_tuned = False
    return ContextBuilder(mock_rag)


@pytest.fixture
def sample_results():
    return [
        {
            'text': 'NIST CSF defines five core functions.',
            'metadata': {'source': 'NIST_CSF.pdf', 'page': 5},
            'content_category': 'definition',
        },
        {
            'text': 'Configure the firewall with these steps.',
            'metadata': {'source': 'Fortinet_Guide.pdf', 'page': 12},
            'content_category': 'procedure',
        },
        {
            'text': 'Example of a SIEM deployment.',
            'metadata': {'source': 'SIEM_Example.pdf', 'page': 3},
            'content_category': 'example',
        },
    ]


class TestBuildContextFromResults:
    def test_builds_context_with_citations(self, builder, sample_results):
        ctx = builder.build_context_from_results(sample_results)
        assert "[Doc - NIST_CSF.pdf p.5]" in ctx
        assert "[Doc - Fortinet_Guide.pdf p.12]" in ctx
        assert "NIST CSF defines five core functions." in ctx

    def test_empty_results_returns_empty(self, builder):
        assert builder.build_context_from_results([]) == ''

    def test_none_results_returns_empty(self, builder):
        assert builder.build_context_from_results(None) == ''

    def test_missing_metadata_uses_defaults(self, builder):
        results = [{'text': 'No metadata here'}]
        ctx = builder.build_context_from_results(results)
        assert "[Doc - Unknown p.0]" in ctx

    def test_results_separated_by_double_newline(self, builder, sample_results):
        ctx = builder.build_context_from_results(sample_results)
        # Each chunk should be separated by \n\n
        assert ctx.count("\n\n") >= 2


class TestBuildStructuredContext:
    def test_categorizes_results(self, builder, sample_results):
        ctx = builder.build_structured_context(sample_results)
        assert 'DEFINICIONES' in ctx
        assert 'PROCEDIMIENTOS' in ctx
        assert 'EJEMPLOS' in ctx

    def test_numbers_citations_sequentially(self, builder, sample_results):
        ctx = builder.build_structured_context(sample_results)
        assert "[Doc 1 -" in ctx
        assert "[Doc 2 -" in ctx
        assert "[Doc 3 -" in ctx

    def test_includes_summary(self, builder, sample_results):
        ctx = builder.build_structured_context(sample_results)
        assert "[RESUMEN]" in ctx

    def test_empty_results_returns_empty(self, builder):
        assert builder.build_structured_context([]) == ""

    def test_truncates_at_max_chars(self, builder):
        results = [
            {'text': 'X' * 500, 'metadata': {'source': 'A.pdf', 'page': 1}, 'content_category': 'definition'},
        ]
        ctx = builder.build_structured_context(results, max_chars=50)
        assert "truncado" in ctx

    def test_default_category_is_mention(self, builder):
        results = [
            {'text': 'Some mention', 'metadata': {'source': 'A.pdf', 'page': 1}},
        ]
        ctx = builder.build_structured_context(results)
        assert 'MENCIONES' in ctx


class TestBuildFocusPrompt:
    def test_includes_focus_header(self, builder):
        prompt = builder.build_focus_prompt("Que es NIST CSF?", "definicion", ["NIST"], "normal")
        assert "FOCUS:" in prompt

    def test_includes_attribute(self, builder):
        prompt = builder.build_focus_prompt("Que es NIST?", "version", ["NIST"], "normal")
        assert "version" in prompt

    def test_includes_entity(self, builder):
        prompt = builder.build_focus_prompt("Que es NIST?", "definicion", ["NIST"], "normal")
        assert "NIST" in prompt

    def test_includes_citation_instruction(self, builder):
        prompt = builder.build_focus_prompt("Que es NIST?", "definicion", ["NIST"], "normal")
        assert "[Doc" in prompt or "cita" in prompt.lower()

    def test_short_mode_adds_brevity_instruction(self, builder):
        prompt = builder.build_focus_prompt("Que es NIST?", "definicion", ["NIST"], "short")
        assert "CORTA" in prompt or "PRECISA" in prompt

    def test_no_entities_still_builds_prompt(self, builder):
        prompt = builder.build_focus_prompt("Que es seguridad?", "definicion", [], "normal")
        assert "FOCUS:" in prompt

    def test_fallback_on_error(self, builder):
        prompt = builder.build_focus_prompt(None, None, None, None)
        assert "FOCUS" in prompt
        assert "INSUFICIENTE" in prompt


class TestCollectSnippets:
    def test_collects_snippets_with_labels(self, builder, sample_results):
        snippets = builder.collect_snippets_for_llm_scoring(sample_results, entities=[], top_n=12)
        assert len(snippets) == 3
        for s in snippets:
            assert 'i' in s
            assert 'label' in s
            assert 'text' in s

    def test_labels_include_doc_number_and_source(self, builder, sample_results):
        snippets = builder.collect_snippets_for_llm_scoring(sample_results, entities=[], top_n=12)
        assert "[Doc 1 -" in snippets[0]['label']
        assert "NIST_CSF" in snippets[0]['label']

    def test_truncates_long_text(self, builder):
        results = [
            {'text': 'X' * 1000, 'metadata': {'source': 'A.pdf', 'page': 1}},
        ]
        snippets = builder.collect_snippets_for_llm_scoring(results, entities=[], top_n=12)
        assert len(snippets[0]['text']) <= 601  # 600 + ellipsis

    def test_respects_top_n(self, builder, sample_results):
        snippets = builder.collect_snippets_for_llm_scoring(sample_results, entities=[], top_n=2)
        assert len(snippets) == 2

    def test_empty_results_returns_empty(self, builder):
        snippets = builder.collect_snippets_for_llm_scoring([], entities=[], top_n=12)
        assert snippets == []
