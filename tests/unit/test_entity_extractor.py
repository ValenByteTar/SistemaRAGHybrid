"""
Tests unitarios para EntityExtractor
"""

import pytest
from rag.entity_extractor import EntityExtractor


class TestEntityExtractor:
    """Tests para EntityExtractor"""
    
    @pytest.fixture
    def extractor(self):
        """Fixture que retorna una instancia de EntityExtractor"""
        return EntityExtractor()
    
    def test_extract_simple_entity(self, extractor):
        """Test extraccion de entidad simple"""
        query = "Que controles define el NIST CSF?"
        entities = extractor.extract_entities(query)
        
        assert any('nist' in e.lower() for e in entities), f"Deberia detectar 'nist', encontro: {entities}"
    
    def test_extract_compound_entity(self, extractor):
        """Test extraccion de entidad compuesta"""
        query = "Informacion sobre certificacion Fortinet NSE4"
        entities = extractor.extract_entities(query)
        
        assert any('nse' in e.lower() or 'fortinet' in e.lower() for e in entities), f"Deberia detectar NSE4/Fortinet, encontro: {entities}"
    
    def test_extract_multiple_entities(self, extractor):
        """Test extraccion de multiples entidades"""
        query = "Diferencias entre CISSP y CEH"
        entities = extractor.extract_entities(query)
        
        assert any('cissp' in e.lower() for e in entities), f"Deberia detectar 'cissp', encontro: {entities}"
        assert any('ceh' in e.lower() for e in entities), f"Deberia detectar 'ceh', encontro: {entities}"
    
    def test_extract_brand_names(self, extractor):
        """Test extraccion de vendors de ciberseguridad"""
        queries = [
            ("Firewall Fortinet", "fortinet"),
            ("Endpoint CrowdStrike", "crowdstrike"),
            ("SIEM Splunk", "splunk"),
        ]
        
        for query, expected in queries:
            entities = extractor.extract_entities(query)
            assert any(expected in e.lower() for e in entities), f"Query '{query}' deberia detectar '{expected}', encontro: {entities}"
    
    def test_extract_certification_codes(self, extractor):
        """Test extraccion de codigos de certificacion"""
        query = "Requisitos para aprobar NSE4"
        entities = extractor.extract_entities(query)
        
        assert any('nse' in e.lower() for e in entities), f"Deberia detectar 'NSE4', encontro: {entities}"
    
    def test_ignore_stopwords(self, extractor):
        """Test que ignora stopwords"""
        query = "Cuantos controles tiene el framework ISO 27001?"
        entities = extractor.extract_entities(query)
        
        stopwords = ['cuantos', 'tiene', 'el']
        for sw in stopwords:
            assert sw not in entities, f"No deberia incluir stopword '{sw}', encontro: {entities}"
    
    def test_extract_doc_reference(self, extractor):
        """Test extraccion de referencia a documento"""
        queries = [
            ("[Doc 3 - NIST CSF v2 p.5]", "NIST CSF v2", 5),
            ("[Doc 1 - ISO 27001 pág.12]", "ISO 27001", 12),
        ]
        
        for query, expected_doc, expected_page in queries:
            result = extractor.extract_doc_reference(query)
            assert result is not None, f"Deberia detectar referencia en '{query}'"
            assert result['doc_name'] == expected_doc, f"Doc name deberia ser '{expected_doc}', es '{result['doc_name']}'"
            assert result['page'] == expected_page, f"Page deberia ser {expected_page}, es {result['page']}"
    
    def test_extract_doc_reference_none(self, extractor):
        """Test cuando no hay referencia a documento"""
        query = "Cuantos controles tiene el NIST CSF?"
        result = extractor.extract_doc_reference(query)
        
        assert result is None, "No deberia detectar referencia"
    
    def test_extract_doc_scope(self, extractor):
        """Test extraccion de scope de documento"""
        queries = [
            ('Busca en "NIST CSF v2.pdf"', "NIST CSF v2.pdf"),
        ]
        
        for query, expected in queries:
            result = extractor.extract_doc_scope(query)
            assert expected.lower() in result.lower(), f"Query '{query}' deberia detectar '{expected}', encontro: '{result}'"
    
    def test_extract_doc_scope_none(self, extractor):
        """Test cuando no hay scope de documento"""
        query = "Cuantos controles tiene el NIST CSF?"
        result = extractor.extract_doc_scope(query)
        
        assert result == '', "No deberia detectar scope"
    
    def test_extract_doc_pages_hint(self, extractor):
        """Test extraccion de paginas especificas"""
        queries = [
            ('pagina 3 y 4 del documento "NIST CSF v2.pdf"', "NIST CSF v2.pdf", [3, 4]),
        ]
        
        for query, expected_doc, expected_pages in queries:
            result = extractor.extract_doc_pages_hint(query)
            assert result is not None, f"Deberia detectar hint en '{query}'"
            assert expected_doc.lower() in result['doc'].lower(), f"Doc deberia contener '{expected_doc}'"
            assert result['pages'] == expected_pages, f"Pages deberia ser {expected_pages}, es {result['pages']}"
    
    def test_extract_doc_pages_hint_none(self, extractor):
        """Test cuando no hay hint de paginas"""
        query = "Cuantos controles tiene el NIST CSF?"
        result = extractor.extract_doc_pages_hint(query)
        
        assert result is None, "No deberia detectar hint"
    
    def test_with_entity_map(self):
        """Test con mapa de entidades de ciberseguridad"""
        entity_map = {
            'nse4': ('Fortinet NSE4', 'NSE4-FortiGate.pdf'),
            'nist csf': ('NIST Cybersecurity Framework', 'NIST-CSF-v2.pdf'),
        }
        
        extractor = EntityExtractor(domain_map=entity_map)
        query = "Informacion del NSE4"
        entities = extractor.extract_entities(query)
        
        assert any('nse' in e.lower() for e in entities), f"Deberia detectar 'nse4', encontro: {entities}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
