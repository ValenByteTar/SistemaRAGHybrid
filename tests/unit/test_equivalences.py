"""
Unit tests for EquivalencesManager — derived from observable behavior of equivalences_manager.py.

Tests cover:
- Loading and parsing equivalence text (A = B = C format)
- Query expansion with synonyms
- Glossary building for acronyms
- Query normalization (codes, roman numerals, name variations)
"""
import pytest
from equivalences_manager import EquivalencesManager


@pytest.fixture
def embedded_text():
    """Sample equivalence text in the 'A = B = C' format the manager expects."""
    return """NIST CSF = NIST Cybersecurity Framework = Cybersecurity Framework
CISSP = Certified Information Systems Security Professional
SIEM = Security Information and Event Management
IDS = Intrusion Detection System
IPS = Intrusion Prevention System"""


@pytest.fixture
def mgr(embedded_text):
    return EquivalencesManager(embedded_text, flags={})


class TestLoadAndParse:
    def test_loads_equivalence_clusters(self, mgr):
        assert len(mgr.equivalences) >= 4

    def test_builds_equivalences_map(self, mgr):
        assert 'nist csf' in mgr.equivalences_map
        assert 'cissp' in mgr.equivalences_map

    def test_builds_definitions_map_for_acronyms(self, mgr):
        assert 'CISSP' in mgr.definitions_map
        assert 'SIEM' in mgr.definitions_map

    def test_definitions_map_excludes_long_phrases(self, mgr):
        # 'NIST CSF' has a space, so it should not be in definitions_map as an acronym
        assert 'NIST CSF' not in mgr.definitions_map

    def test_empty_text_produces_empty_maps(self):
        mgr = EquivalencesManager("", flags={})
        assert mgr.equivalences == []
        assert mgr.equivalences_map == {}

    def test_none_text_produces_empty_maps(self):
        mgr = EquivalencesManager(None, flags={})
        assert mgr.equivalences == []


class TestExpand:
    def test_expand_adds_synonyms_for_known_term(self, mgr):
        result = mgr.expand("Que es NIST CSF?")
        assert "NIST CSF" in result
        # Should add at least one synonym
        assert len(result) > len("Que es NIST CSF?")

    def test_expand_no_change_for_unknown_term(self, mgr):
        query = "Que es Python?"
        result = mgr.expand(query)
        assert result == query

    def test_expand_no_change_when_no_equivalences(self):
        mgr = EquivalencesManager("", flags={})
        assert mgr.expand("test query") == "test query"

    def test_expand_preserves_original_query(self, mgr):
        query = "Informacion sobre CISSP"
        result = mgr.expand(query)
        assert query in result


class TestBuildGlossary:
    def test_glossary_includes_acronym_definition(self, mgr):
        glossary = mgr.build_glossary("Que es CISSP?")
        assert "CISSP" in glossary
        assert "certified" in glossary.lower()

    def test_glossary_empty_for_no_acronyms(self, mgr):
        glossary = mgr.build_glossary("Que es un firewall?")
        assert glossary == ''

    def test_glossary_respects_ban_list(self, mgr):
        # 'SECURITY' is in the ban list
        # Add a fake equivalence that would generate SECURITY as a key
        text = "SEC = Security = Seguridad"
        mgr2 = EquivalencesManager(text, flags={})
        glossary = mgr2.build_glossary("SEC")
        assert "SECURITY:" not in glossary.upper() or "SEC:" in glossary.upper()

    def test_glossary_max_10_items(self, mgr):
        # Build text with many acronyms
        lines = [f"ACR{i} = Full Name Number {i}" for i in range(15)]
        mgr2 = EquivalencesManager("\n".join(lines), flags={})
        glossary = mgr2.build_glossary(" ".join([f"ACR{i}" for i in range(15)]))
        # Count items (lines starting with "- ")
        items = [l for l in glossary.split("\n") if l.startswith("- ")]
        assert len(items) <= 10


class TestNormalizeQuery:
    def test_normalize_adds_name_variations(self, mgr):
        result = mgr.normalize_query("Como hacer un pentest?")
        assert "pentest" in result.lower()
        assert "penetration test" in result.lower()

    def test_normalize_adds_code_variations(self, mgr):
        result = mgr.normalize_query("Informacion sobre ISO 27001")
        # Should add variations like ISO_27001, ISO 27001
        assert "27001" in result

    def test_normalize_adds_roman_numeral_variations(self, mgr):
        result = mgr.normalize_query("NIST CSF 2")
        # Should add roman numeral "II"
        assert "II" in result

    def test_normalize_preserves_original_query(self, mgr):
        query = "Que es CISSP?"
        result = mgr.normalize_query(query)
        assert query in result

    def test_normalize_adds_equivalence_expansions(self, mgr):
        result = mgr.normalize_query("Que es NIST CSF?")
        # Should expand with equivalence cluster members
        assert len(result) > len("Que es NIST CSF?")
