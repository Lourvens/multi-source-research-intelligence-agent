"""
Tests for document loader module.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from langchain_classic.schema import Document

from src.ingestion.document_loader import DocumentLoader
from src.constants import (
    CONTENT_TYPE_ABSTRACT_ONLY,
    CONTENT_TYPE_FULL_PAPER,
    DATA_SOURCE_ARXIV
)


class TestDocumentLoader:
    """Test DocumentLoader class."""
    
    def test_init(self):
        """Test DocumentLoader initialization."""
        loader = DocumentLoader()
        assert loader.metadata_dir is not None
        assert loader.pdf_processor is not None
    
    def test_load_metadata_success(self, temp_data_dir, sample_arxiv_metadata, monkeypatch):
        """Test loading metadata from file."""
        from src import config
        original_raw_dir = config.RAW_DATA_DIR
        monkeypatch.setattr(config, "RAW_DATA_DIR", temp_data_dir["data"] / "raw")
        
        # Create metadata file (metadata_dir is RAW_DATA_DIR / "arxiv_metadata")
        metadata_file = temp_data_dir["raw"] / f"{sample_arxiv_metadata['id']}.json"
        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        with metadata_file.open("w", encoding="utf-8") as f:
            json.dump(sample_arxiv_metadata, f)
        
        try:
            loader = DocumentLoader()
            metadata = loader.load_metadata(sample_arxiv_metadata["id"])
            
            assert metadata is not None
            assert metadata["id"] == sample_arxiv_metadata["id"]
            assert metadata["title"] == sample_arxiv_metadata["title"]
        finally:
            monkeypatch.setattr(config, "RAW_DATA_DIR", original_raw_dir)
    
    def test_load_metadata_not_found(self, temp_data_dir, monkeypatch):
        """Test loading non-existent metadata."""
        from src import config
        original_raw_dir = config.RAW_DATA_DIR
        monkeypatch.setattr(config, "RAW_DATA_DIR", temp_data_dir["data"] / "raw")
        
        try:
            loader = DocumentLoader()
            metadata = loader.load_metadata("nonexistent_id")
            assert metadata is None
        finally:
            monkeypatch.setattr(config, "RAW_DATA_DIR", original_raw_dir)
    
    def test_create_abstract_document(self, sample_arxiv_metadata):
        """Test creating abstract-only document."""
        loader = DocumentLoader()
        doc = loader._create_abstract_document(
            sample_arxiv_metadata["id"],
            sample_arxiv_metadata
        )
        
        assert isinstance(doc, Document)
        assert sample_arxiv_metadata["title"] in doc.page_content
        assert sample_arxiv_metadata["summary"] in doc.page_content
        assert doc.metadata["arxiv_id"] == sample_arxiv_metadata["id"]
        assert doc.metadata["content_type"] == CONTENT_TYPE_ABSTRACT_ONLY
        assert doc.metadata["source"] == DATA_SOURCE_ARXIV
    
    @patch('src.ingestion.document_loader.PDFProcessor')
    def test_load_all_documents_abstracts_only(
        self,
        mock_pdf_processor_class,
        temp_data_dir,
        sample_arxiv_metadata,
        monkeypatch
    ):
        """Test loading documents with abstracts only."""
        from src import config
        original_raw_dir = config.RAW_DATA_DIR
        monkeypatch.setattr(config, "RAW_DATA_DIR", temp_data_dir["data"] / "raw")
        
        # Create metadata file (metadata_dir is RAW_DATA_DIR / "arxiv_metadata")
        metadata_file = temp_data_dir["raw"] / f"{sample_arxiv_metadata['id']}.json"
        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        with metadata_file.open("w", encoding="utf-8") as f:
            json.dump(sample_arxiv_metadata, f)
        
        # Mock PDF processor
        mock_pdf_processor = Mock()
        mock_pdf_processor.process_all_pdfs.return_value = {}
        mock_pdf_processor_class.return_value = mock_pdf_processor
        
        try:
            loader = DocumentLoader()
            documents = loader.load_all_documents(
                include_full_text=False,
                max_documents=1
            )
            
            assert len(documents) == 1
            assert documents[0].metadata["content_type"] == CONTENT_TYPE_ABSTRACT_ONLY
            mock_pdf_processor.process_all_pdfs.assert_not_called()
        finally:
            monkeypatch.setattr(config, "RAW_DATA_DIR", original_raw_dir)
    
    def test_load_all_documents_no_metadata(self, temp_data_dir, monkeypatch):
        """Test loading when no metadata files exist."""
        from src import config
        original_raw_dir = config.RAW_DATA_DIR
        monkeypatch.setattr(config, "RAW_DATA_DIR", temp_data_dir["data"] / "raw")
        
        try:
            loader = DocumentLoader()
            documents = loader.load_all_documents()
            
            assert len(documents) == 0
        finally:
            monkeypatch.setattr(config, "RAW_DATA_DIR", original_raw_dir)
    
    def test_save_loaded_documents(self, temp_data_dir, sample_document, monkeypatch):
        """Test saving loaded documents to disk."""
        from src import config
        original_processed_dir = config.PROCESSED_DATA_DIR
        monkeypatch.setattr(config, "PROCESSED_DATA_DIR", temp_data_dir["data"] / "processed")
        
        try:
            loader = DocumentLoader()
            saved_paths = loader._save_loaded_documents(
                [sample_document],
                include_full_text=False
            )
            
            assert len(saved_paths) == 1
            assert saved_paths[0].exists()
            
            # Verify file content (each document is saved as a separate file)
            with saved_paths[0].open("r", encoding="utf-8") as f:
                data = json.load(f)
            
            assert "metadata" in data
            assert "page_content" in data
            assert data["metadata"]["arxiv_id"] == sample_document.metadata["arxiv_id"]
            assert data["page_content"] == sample_document.page_content
        finally:
            monkeypatch.setattr(config, "PROCESSED_DATA_DIR", original_processed_dir)


