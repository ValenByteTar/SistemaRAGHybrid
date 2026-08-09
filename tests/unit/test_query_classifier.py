"""
Unit tests for QueryClassifier — derived from observable behavior of query_classifier.py.

Tests cover:
- Out-of-domain detection
- Detailed query detection
- Multi-document detection
- Comparison detection
- Aggregation detection
- Listing detection
- Simple numeric detection
- Follow-up detection
- Conceptual/procedural detection
- Tech/vendor filter extraction
"""
import pytest
from query_classifier import QueryClassifier


@pytest.fixture
def classifier():
    """Classifier with no flags and a simple entity extractor stub."""
    return QueryClassifier(flags={}, extract_entities_fn=lambda q: [])


@pytest.fixture
def classifier_with_entities():
    """Classifier whose entity extractor returns specific entities."""
    def extract(q):
        q_lower = q.lower()
        if 'cissp' in q_lower:
            return ['CISSP']
        if 'iso' in q_lower:
            return ['ISO']
        if 'nist' in q_lower:
            return ['NIST']
        return []
    return QueryClassifier(flags={}, extract_entities_fn=extract)


class TestIsOutOfDomain:
    def test_video_game_is_out_of_domain(self, classifier):
        assert classifier.is_out_of_domain("Como ganar en Minecraft?") is True

    def test_cybersecurity_is_in_domain(self, classifier):
        assert classifier.is_out_of_domain("Que es el NIST Cybersecurity Framework?") is False

    def test_cooking_is_out_of_domain(self, classifier):
        assert classifier.is_out_of_domain("Como hornear pan de harina?") is True

    def test_mixed_domain_and_ood_stays_in_domain(self, classifier):
        assert classifier.is_out_of_domain("Que es un firewall de Minecraft?") is False

    def test_empty_query_not_out_of_domain(self, classifier):
        assert classifier.is_out_of_domain("") is False

    def test_neutral_query_not_out_of_domain(self, classifier):
        assert classifier.is_out_of_domain("Que es un framework?") is False


class TestIsDetailed:
    def test_toda_informacion_is_detailed(self, classifier):
        assert classifier.is_detailed("Dame toda la informacion sobre NIST CSF") is True

    def test_completa_is_detailed(self, classifier):
        assert classifier.is_detailed("Descripcion completa del CISSP") is True

    def test_simple_question_not_detailed(self, classifier):
        assert classifier.is_detailed("Que es NIST?") is False

    def test_min_word_count_is_detailed(self, classifier):
        assert classifier.is_detailed("Responde con minimo 200 palabras sobre ISO 27001") is True


class TestIsMultiDocument:
    def test_todos_is_multi_document(self, classifier):
        assert classifier.is_multi_document("Lista todos los frameworks de seguridad") is True

    def test_comparar_is_multi_document(self, classifier):
        assert classifier.is_multi_document("Comparar NIST e ISO") is True

    def test_single_entity_not_multi(self, classifier):
        # "que" is in multi_keywords, so any "que" query is multi-doc by design
        assert classifier.is_multi_document("Define seguridad.") is False


class TestIsComparison:
    def test_compara_con_is_comparison(self, classifier):
        assert classifier.is_comparison("Compara NIST CSF con ISO 27001") is True

    def test_vs_is_comparison(self, classifier):
        assert classifier.is_comparison("CISSP vs CEH") is True

    def test_no_comparison_word_not_comparison(self, classifier):
        assert classifier.is_comparison("Que es NIST CSF?") is False

    def test_comparison_word_without_connector_not_comparison(self, classifier):
        assert classifier.is_comparison("Compara frameworks") is False


class TestIsAggregation:
    def test_total_requisitos_is_aggregation(self, classifier_with_entities):
        # Without specific entities, aggregation should be True
        clf = QueryClassifier(flags={}, extract_entities_fn=lambda q: [])
        assert clf.is_aggregation("Cual es el total de requisitos de PCI DSS?") is True

    def test_specific_info_not_aggregation(self, classifier):
        assert classifier.is_aggregation("Informacion sobre NIST CSF") is False

    def test_listing_not_aggregation(self, classifier):
        assert classifier.is_aggregation("Lista todos los frameworks") is False


class TestIsListing:
    def test_listado_frameworks_is_listing(self, classifier):
        assert classifier.is_listing("Listado completo de frameworks") is True

    def test_listar_controles_is_listing(self, classifier):
        assert classifier.is_listing("Listar todos los controles de ISO 27001") is True

    def test_no_target_not_listing(self, classifier):
        assert classifier.is_listing("Lista de cosas") is False


class TestIsSimpleNumeric:
    def test_cuantos_with_entity_is_numeric(self, classifier_with_entities):
        assert classifier_with_entities.is_simple_numeric("Cuantos dominios tiene CISSP?") is True

    def test_cuantos_without_entity_not_numeric(self, classifier_with_entities):
        assert classifier_with_entities.is_simple_numeric("Cuantos?") is False

    def test_comparison_not_numeric(self, classifier_with_entities):
        # is_simple_numeric calls is_comparison internally
        result = classifier_with_entities.is_simple_numeric("Cuantos controles tiene CISSP vs CEH?")
        # The comparison check should prevent numeric classification
        assert result is False


class TestIsFollowUp:
    def test_y_prefix_is_followup(self, classifier):
        assert classifier.is_follow_up("Y su ubicacion?") is True

    def test_short_anaphoric_is_followup(self, classifier):
        assert classifier.is_follow_up("este framework") is True

    def test_long_standalone_not_followup(self, classifier):
        assert classifier.is_follow_up("Que es el NIST Cybersecurity Framework?") is False


class TestIsConceptual:
    def test_como_funciona_is_conceptual(self, classifier):
        # "como funciona una" is a conceptual keyword; lowercase "como" won't match
        # the specific_names regex ([A-Z][a-z]+), so this is conceptual
        assert classifier.is_conceptual("como funciona una vpn?") is True

    def test_que_es_una_is_conceptual(self, classifier):
        # lowercase "que es un" is a conceptual keyword; no capitalized names
        assert classifier.is_conceptual("que es un firewall?") is True

    def test_with_specific_name_not_conceptual(self, classifier):
        # "Cisco" matches the specific_names regex, so not conceptual
        assert classifier.is_conceptual("Como funciona Cisco ASA?") is False

    def test_capitalized_starting_word_not_conceptual(self, classifier):
        # "Como" (capitalized) matches specific_names regex, blocking conceptual
        assert classifier.is_conceptual("Como funciona una VPN?") is False


class TestIsProcedural:
    def test_como_hacer_is_procedural(self, classifier):
        assert classifier.is_procedural("Como configurar un firewall?") is True

    def test_pasos_para_is_procedural(self, classifier):
        assert classifier.is_procedural("Pasos para implementar ISO 27001") is True

    def test_what_is_not_procedural(self, classifier):
        assert classifier.is_procedural("Que es NIST CSF?") is False


class TestIsSpecificCount:
    def test_cuantos_controles_is_specific_count(self, classifier):
        assert classifier.is_specific_count("Cuantos controles tiene ISO 27001?") is True

    def test_cuantos_dominios_is_specific_count(self, classifier):
        assert classifier.is_specific_count("Cuantos dominios tiene CISSP?") is True

    def test_version_not_specific_count(self, classifier):
        assert classifier.is_specific_count("Que version de PCI DSS existe?") is False


class TestExtractTechFilter:
    def test_firewall_maps_to_network_category(self, classifier):
        # "firewall" is in the Network category list
        result = classifier.extract_tech_filter("Que firewall usar?")
        assert result == 'Network'

    def test_cloud_keyword_maps_to_cloud(self, classifier):
        result = classifier.extract_tech_filter("Configurar AWS")
        assert result == 'Cloud'

    def test_identity_keyword_maps_to_identity(self, classifier):
        result = classifier.extract_tech_filter("Configurar MFA")
        assert result == 'Identity'

    def test_no_tech_returns_empty(self, classifier):
        result = classifier.extract_tech_filter("Que es seguridad?")
        assert result == ''


class TestExtractVendorFilter:
    def test_extracts_vendor(self, classifier):
        result = classifier.extract_vendor_filter("Configurar Fortinet firewall")
        assert 'fortinet' in result.lower()

    def test_no_vendor_returns_empty(self, classifier):
        result = classifier.extract_vendor_filter("Que es un firewall?")
        assert result == ''
