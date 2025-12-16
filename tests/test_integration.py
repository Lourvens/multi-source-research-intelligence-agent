"""
Integration tests for Phase 1 pipeline.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch

from langchain_classic.schema import Document

from src.embedding.document_processor import process_arxiv_abstracts
from src.constants import DATA_SOURCE_ARXIV, CONTENT_TYPE_ABSTRACT_ONLY


class TestPhase1Integration:
    """Integration tests for Phase 1 complete pipeline."""
    
    @patch('src.embedding.document_processor.DocumentLoader')
    def test_end_to_end_pipeline(self, mock_loader_class, sample_document):
        """Test complete pipeline: Load → Chunk → Embed."""
        # Setup mock loader
        mock_loader = Mock()
        mock_loader.load_all_documents.return_value = [sample_document]
        mock_loader_class.return_value = mock_loader
        
        # Run pipeline
        result = process_arxiv_abstracts(max_documents=1)
        
        # Verify result
        assert len(result) == 1
        assert isinstance(result[0], Document)
        assert "embedding" in result[0].metadata
        
        # Verify metadata preservation
        assert result[0].metadata["arxiv_id"] == sample_document.metadata["arxiv_id"]
        assert result[0].metadata["source"] == DATA_SOURCE_ARXIV
    
    @patch('src.embedding.document_processor.DocumentLoader')
    def test_pipeline_metadata_completeness(self, mock_loader_class, sample_document):
        """Test that pipeline preserves all required metadata (Rule 2)."""
        mock_loader = Mock()
        mock_loader.load_all_documents.return_value = [sample_document]
        mock_loader_class.return_value = mock_loader
        
        result = process_arxiv_abstracts(max_documents=1)
        
        doc = result[0]
        metadata = doc.metadata
        
        # Required metadata fields (per AGENTS.md Rule 2)
        assert "source" in metadata
        assert "arxiv_id" in metadata  # paper_id
        assert "title" in metadata
        assert "authors" in metadata
        assert "published" in metadata
        assert "pdf_url" in metadata
        assert "embedding" in metadata  # Added by embedder
    
    @patch('src.embedding.document_processor.DocumentLoader')
    def test_pipeline_empty_input(self, mock_loader_class):
        """Test pipeline with empty input."""
        mock_loader = Mock()
        mock_loader.load_all_documents.return_value = []
        mock_loader_class.return_value = mock_loader
        
        result = process_arxiv_abstracts()
        assert len(result) == 0
    
    @patch('src.embedding.document_processor.DocumentLoader')
    @patch('src.embedding.document_processor.save_processed_chunks')
    def test_pipeline_with_saving(
        self,
        mock_save_chunks,
        mock_loader_class,
        sample_document
    ):
        """Test pipeline with save_to_disk enabled."""
        mock_loader = Mock()
        mock_loader.load_all_documents.return_value = [sample_document]
        mock_loader_class.return_value = mock_loader
        
        from src.embedding.document_processor import DocumentProcessor
        
        processor = DocumentProcessor()
        result = processor.process_documents(
            max_documents=1,
            save_to_disk=True
        )
        
        assert len(result) == 1
        mock_save_chunks.assert_called_once()
    
    def test_constants_consistency(self):
        """Test that constants are used consistently across modules."""
        from src.constants import (
            ChunkingStrategy,
            ContentType,
            PDFLoaderType,
            DataSource
        )
        
        # Verify enums have expected values
        assert ChunkingStrategy.NONE.value == "none"
        assert ContentType.ABSTRACT_ONLY.value == "abstract_only"
        assert PDFLoaderType.PYMUPDF.value == "pymupdf"
        assert DataSource.ARXIV.value == "arxiv"
    
    @patch('src.embedding.document_processor.DocumentLoader')
    def test_pipeline_idempotency(self, mock_loader_class, sample_document):
        """Test that running pipeline twice produces same results (Rule 3)."""
        mock_loader = Mock()
        mock_loader.load_all_documents.return_value = [sample_document]
        mock_loader_class.return_value = mock_loader
        
        # Run twice
        result1 = process_arxiv_abstracts(max_documents=1)
        result2 = process_arxiv_abstracts(max_documents=1)
        
        # Results should be identical
        assert len(result1) == len(result2)
        assert result1[0].page_content == result2[0].page_content
        assert result1[0].metadata["arxiv_id"] == result2[0].metadata["arxiv_id"]

