"""
Unit tests for TextChunker — derived from observable behavior of src/chunker.py.

Tests cover:
- Semantic chunking (character-based, paragraph/sentence boundaries)
- Metadata preservation (source, page, chunk_index)
- Edge cases (empty text, single paragraph, oversized paragraph)
"""
import pytest
from chunker import TextChunker


class TestSplitTextSemantic:
    """Tests for split_text_semantic (character-based chunking)."""

    def test_single_short_paragraph_returns_one_chunk(self):
        chunker = TextChunker(chunk_size=800, overlap=200)
        text = "This is a short paragraph about NIST CSF."
        chunks = chunker.split_text_semantic(text)
        assert len(chunks) == 1
        assert "NIST CSF" in chunks[0]

    def test_multiple_paragraphs_under_chunk_size_merged(self):
        chunker = TextChunker(chunk_size=800, overlap=200)
        text = "Paragraph one about ISO 27001.\n\nParagraph two about CISSP."
        chunks = chunker.split_text_semantic(text)
        assert len(chunks) == 1
        assert "ISO 27001" in chunks[0]
        assert "CISSP" in chunks[0]

    def test_paragraphs_exceeding_chunk_size_split(self):
        chunker = TextChunker(chunk_size=50, overlap=10)
        text = "Short para A.\n\n" + "B" * 80
        chunks = chunker.split_text_semantic(text)
        assert len(chunks) >= 2

    def test_empty_text_returns_empty_list(self):
        chunker = TextChunker(chunk_size=800, overlap=200)
        chunks = chunker.split_text_semantic("")
        assert chunks == []

    def test_only_whitespace_returns_empty_list(self):
        chunker = TextChunker(chunk_size=800, overlap=200)
        chunks = chunker.split_text_semantic("   \n\n   \n\n   ")
        assert chunks == []

    def test_oversized_paragraph_split_by_sentences(self):
        chunker = TextChunker(chunk_size=60, overlap=10)
        para = "Sentence one is here. Sentence two follows. Sentence three ends it."
        text = para
        chunks = chunker.split_text_semantic(text)
        assert len(chunks) >= 2

    def test_strips_whitespace_from_chunks(self):
        chunker = TextChunker(chunk_size=800, overlap=200)
        text = "  Some content with surrounding whitespace.  \n\n  More content.  "
        chunks = chunker.split_text_semantic(text)
        for c in chunks:
            assert c == c.strip()


class TestCreateChunksWithMetadata:
    """Tests for create_chunks_with_metadata — the full chunking pipeline."""

    @pytest.fixture
    def chunker(self):
        return TextChunker(chunk_size=200, overlap=50, token_chunking=False)

    @pytest.fixture
    def pdf_data(self):
        return {
            'filename': 'NIST_CSF_v2.pdf',
            'filepath': '/docs/NIST_CSF_v2.pdf',
            'success': True,
            'doc_date': '2024-02-15',
            'category': 'framework',
            'pages': [
                {
                    'page_num': 1,
                    'text': 'The NIST Cybersecurity Framework is a set of guidelines.',
                    'section': 'Introduction',
                },
                {
                    'page_num': 2,
                    'text': 'It defines five core functions: Identify, Protect, Detect, Respond, Recover.',
                    'section': 'Core Functions',
                },
            ],
        }

    def test_returns_chunks_with_metadata(self, chunker, pdf_data):
        chunks = chunker.create_chunks_with_metadata(pdf_data)
        assert len(chunks) >= 2
        for c in chunks:
            assert 'id' in c
            assert 'text' in c
            assert 'metadata' in c

    def test_metadata_preserves_source_filename(self, chunker, pdf_data):
        chunks = chunker.create_chunks_with_metadata(pdf_data)
        for c in chunks:
            assert c['metadata']['source'] == 'NIST_CSF_v2.pdf'

    def test_metadata_preserves_page_number(self, chunker, pdf_data):
        chunks = chunker.create_chunks_with_metadata(pdf_data)
        pages = {c['metadata']['page'] for c in chunks}
        assert 1 in pages
        assert 2 in pages

    def test_metadata_preserves_filepath(self, chunker, pdf_data):
        chunks = chunker.create_chunks_with_metadata(pdf_data)
        for c in chunks:
            assert c['metadata']['filepath'] == '/docs/NIST_CSF_v2.pdf'

    def test_chunk_ids_are_unique(self, chunker, pdf_data):
        chunks = chunker.create_chunks_with_metadata(pdf_data)
        ids = [c['id'] for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_index_increments(self, chunker, pdf_data):
        chunks = chunker.create_chunks_with_metadata(pdf_data)
        indices = [c['metadata']['chunk_index'] for c in chunks]
        assert indices == sorted(indices)
        assert indices[0] == 0

    def test_failed_pdf_returns_empty(self, chunker):
        chunks = chunker.create_chunks_with_metadata({'success': False, 'pages': []})
        assert chunks == []

    def test_metadata_preserves_section(self, chunker, pdf_data):
        chunks = chunker.create_chunks_with_metadata(pdf_data)
        sections = {c['metadata']['section'] for c in chunks}
        assert 'Introduction' in sections
        assert 'Core Functions' in sections

    def test_metadata_preserves_doc_date_and_category(self, chunker, pdf_data):
        chunks = chunker.create_chunks_with_metadata(pdf_data)
        for c in chunks:
            assert c['metadata']['doc_date'] == '2024-02-15'
            assert c['metadata']['category'] == 'framework'


class TestChunkerDefaults:
    """Tests for default constructor values."""

    def test_default_chunk_size_is_800(self):
        chunker = TextChunker()
        assert chunker.chunk_size == 800

    def test_default_overlap_is_200(self):
        chunker = TextChunker()
        assert chunker.overlap == 200

    def test_default_token_chunking_is_false(self):
        chunker = TextChunker()
        assert chunker.token_chunking is False

    def test_custom_values_respected(self):
        chunker = TextChunker(chunk_size=350, overlap=50, token_chunking=True, token_chunk_size=350)
        assert chunker.chunk_size == 350
        assert chunker.overlap == 50
        assert chunker.token_chunking is True
        assert chunker.token_chunk_size == 350
