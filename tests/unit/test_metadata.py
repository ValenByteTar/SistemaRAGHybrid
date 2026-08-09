"""
Unit tests for metadata contracts and citation marker parsing.

Tests cover:
- Citation marker extraction [Doc N - source p.X] (from run_cybersec_eval.extract_citation_sources)
- Citation marker building (from context_builder.build_context_from_results)
- Source matching logic (from run_cybersec_eval._src_matches)
- Context builder structured context with [Doc N - source p.X] format
"""
import re
import pytest


# --- Citation extraction (replicates run_cybersec_eval.extract_citation_sources) ---

def extract_citation_sources(answer: str) -> list:
    """Extract source names from [Doc N - name p.X] markers."""
    pattern = r"\[Doc\s*\d+[^\]]*?-\s*([^\]]+?)\s*p\.\d+"
    return re.findall(pattern, answer, re.IGNORECASE)


# --- Source matching (replicates run_cybersec_eval._src_matches) ---

def src_matches(api_name: str, exp_name: str) -> bool:
    """Bidirectional partial match between source names."""
    a, e = api_name.lower(), exp_name.lower()
    return e in a or a in e


class TestCitationExtraction:
    def test_extracts_single_citation(self):
        answer = "NIST CSF defines five functions [Doc 1 - NIST CSF v2.pdf p.5]"
        sources = extract_citation_sources(answer)
        assert sources == ["NIST CSF v2.pdf"]

    def test_extracts_multiple_citations(self):
        answer = "See [Doc 1 - NIST CSF v2.pdf p.5] and [Doc 3 - ISO 27001.pdf p.12]"
        sources = extract_citation_sources(answer)
        assert "NIST CSF v2.pdf" in sources
        assert "ISO 27001.pdf" in sources
        assert len(sources) == 2

    def test_no_citations_returns_empty(self):
        answer = "This is a response without citations."
        sources = extract_citation_sources(answer)
        assert sources == []

    def test_case_insensitive_extraction(self):
        answer = "See [doc 2 - Framework.pdf p.10]"
        sources = extract_citation_sources(answer)
        assert sources == ["Framework.pdf"]

    def test_citation_with_extra_spaces(self):
        answer = "Result from [Doc  5  -  Guide.pdf  p.20]"
        sources = extract_citation_sources(answer)
        assert sources == ["Guide.pdf"]

    def test_citation_with_multi_word_source(self):
        answer = "According to [Doc 1 - NIST Cybersecurity Framework 2.0.pdf p.5]"
        sources = extract_citation_sources(answer)
        assert "NIST Cybersecurity Framework 2.0.pdf" in sources


class TestSourceMatching:
    def test_exact_match(self):
        assert src_matches("NIST CSF v2.pdf", "NIST CSF v2.pdf") is True

    def test_partial_match_api_contains_expected(self):
        assert src_matches("NIST CSF v2 Final.pdf", "NIST CSF v2") is True

    def test_partial_match_expected_contains_api(self):
        assert src_matches("ISO", "ISO 27001.pdf") is True

    def test_no_match(self):
        assert src_matches("CISSP Guide.pdf", "NIST CSF.pdf") is False

    def test_case_insensitive(self):
        assert src_matches("nist csf.pdf", "NIST CSF.PDF") is True


class TestContextBuilderCitations:
    """Tests for context_builder.build_context_from_results citation format."""

    def build_context(self, results):
        """Replicates context_builder.build_context_from_results."""
        parts = []
        for r in results or []:
            md = r.get('metadata', {}) or {}
            src = md.get('source', 'Unknown')
            page = md.get('page', 0)
            prefix = f"[Doc - {src} p.{page}]"
            parts.append(prefix + "\n" + (r.get('text') or ''))
        return "\n\n".join(parts)

    def test_context_includes_citation_prefix(self):
        results = [
            {'text': 'NIST CSF content', 'metadata': {'source': 'NIST_CSF.pdf', 'page': 5}},
        ]
        ctx = self.build_context(results)
        assert "[Doc - NIST_CSF.pdf p.5]" in ctx
        assert "NIST CSF content" in ctx

    def test_context_multiple_results_separated(self):
        results = [
            {'text': 'Content A', 'metadata': {'source': 'A.pdf', 'page': 1}},
            {'text': 'Content B', 'metadata': {'source': 'B.pdf', 'page': 2}},
        ]
        ctx = self.build_context(results)
        assert "[Doc - A.pdf p.1]" in ctx
        assert "[Doc - B.pdf p.2]" in ctx
        assert "Content A" in ctx
        assert "Content B" in ctx

    def test_empty_results_produce_empty_context(self):
        ctx = self.build_context([])
        assert ctx == ''

    def test_none_results_produce_empty_context(self):
        ctx = self.build_context(None)
        assert ctx == ''

    def test_missing_metadata_uses_defaults(self):
        results = [{'text': 'No metadata'}]
        ctx = self.build_context(results)
        assert "[Doc - Unknown p.0]" in ctx


class TestStructuredContextCitations:
    """Tests for context_builder.build_structured_context [Doc N - source p.X] format."""

    def build_structured_context(self, results, max_chars=6000):
        """Replicates context_builder.build_structured_context."""
        if not results:
            return ""
        by_category = {'definition': [], 'procedure': [], 'example': [], 'mention': []}
        for r in results:
            cat = r.get('content_category', 'mention')
            by_category[cat].append(r)
        context_parts = []
        total_chars = 0
        summary_parts = [f"{len(items)} {cat}" for cat, items in by_category.items() if items]
        if summary_parts:
            context_parts.append(f"[RESUMEN] Documentos organizados: {', '.join(summary_parts)}")
        section_names = {
            'definition': '=== DEFINICIONES Y CONCEPTOS ===',
            'procedure': '=== PROCEDIMIENTOS Y MEJORES PRÁCTICAS ===',
            'example': '=== EJEMPLOS Y CASOS ===',
            'mention': '=== MENCIONES ADICIONALES ===',
        }
        doc_counter = 0
        for category in ['definition', 'procedure', 'example', 'mention']:
            items = by_category.get(category, [])
            if not items:
                continue
            section_text = f"\n{section_names[category]}\n"
            if total_chars + len(section_text) > max_chars:
                break
            context_parts.append(section_text)
            total_chars += len(section_text)
            for r in items:
                doc_counter += 1
                source = r.get('metadata', {}).get('source', 'Unknown')
                page = r.get('metadata', {}).get('page', 0)
                text = r.get('text', '')[:700]
                fragment = f"[Doc {doc_counter} - {source[:50]} p.{page}]\n{text}\n"
                if total_chars + len(fragment) > max_chars:
                    context_parts.append("[Contexto truncado por límite de tamaño]")
                    break
                context_parts.append(fragment)
                total_chars += len(fragment)
        return '\n'.join(context_parts)

    def test_structured_context_includes_numbered_citations(self):
        results = [
            {'text': 'A definition', 'metadata': {'source': 'Guide.pdf', 'page': 3}, 'content_category': 'definition'},
        ]
        ctx = self.build_structured_context(results)
        assert "[Doc 1 - Guide.pdf p.3]" in ctx

    def test_structured_context_categorizes_results(self):
        results = [
            {'text': 'Def', 'metadata': {'source': 'A.pdf', 'page': 1}, 'content_category': 'definition'},
            {'text': 'Step', 'metadata': {'source': 'B.pdf', 'page': 2}, 'content_category': 'procedure'},
        ]
        ctx = self.build_structured_context(results)
        assert 'DEFINICIONES' in ctx
        assert 'PROCEDIMIENTOS' in ctx

    def test_structured_context_empty_returns_empty(self):
        assert self.build_structured_context([]) == ""

    def test_structured_context_truncates_at_max_chars(self):
        results = [
            {'text': 'X' * 500, 'metadata': {'source': 'A.pdf', 'page': 1}, 'content_category': 'definition'},
        ]
        ctx = self.build_structured_context(results, max_chars=50)
        assert "truncado" in ctx
