"""
Tests for document processor (full pipeline).
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from langchain_classic.schema import Document

from src.embedding.document_processor import DocumentProcessor, process_arxiv_abstracts
from src.constants import ChunkingStrategy, CHUNK_STRATEGY_NONE


class TestDocumentProcessor:
    """Test DocumentProcessor class."""
    
    def test_init_default(self):
        """Test DocumentProcessor initialization."""
        processor = DocumentProcessor()
        assert processor.loader is not None
        assert processor.chunker is not None
        assert processor.embedder is not None
    
    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        processor = DocumentProcessor(
            embedding_model="all-MiniLM-L6-v2",
            chunk_strategy=CHUNK_STRATEGY_NONE,
            chunk_size=256,
            chunk_overlap=25
        )
        assert processor.chunker.chunk_size == 256
        assert processor.chunker.chunk_overlap == 25
    
    @patch('src.embedding.document_processor.DocumentLoader')
    def test_process_documents_empty(self, mock_loader_class):
        """Test processing when no documents are loaded."""
        mock_loader = Mock()
        mock_loader.load_all_documents.return_value = []
        mock_loader_class.return_value = mock_loader
        
        processor = DocumentProcessor()
        result = processor.process_documents()
        
        assert len(result) == 0
    
    @patch('src.embedding.document_processor.DocumentLoader')
    def test_process_documents_success(self, mock_loader_class, sample_document):
        """Test successful document processing pipeline."""
        mock_loader = Mock()
        mock_loader.load_all_documents.return_value = [sample_document]
        mock_loader_class.return_value = mock_loader
        
        processor = DocumentProcessor(chunk_strategy=CHUNK_STRATEGY_NONE)
        result = processor.process_documents(max_documents=1)
        
        assert len(result) == 1
        assert "embedding" in result[0].metadata
        mock_loader.load_all_documents.assert_called_once()
    
    @patch('src.embedding.document_processor.DocumentLoader')
    @patch('src.embedding.document_processor.save_processed_chunks')
    def test_process_documents_save_to_disk(
        self,
        mock_save_chunks,
        mock_loader_class,
        sample_document
    ):
        """Test that processed chunks are saved to disk."""
        mock_loader = Mock()
        mock_loader.load_all_documents.return_value = [sample_document]
        mock_loader_class.return_value = mock_loader
        
        mock_save_chunks.return_value = Mock()  # Mock saved path
        
        processor = DocumentProcessor(chunk_strategy=CHUNK_STRATEGY_NONE)
        result = processor.process_documents(save_to_disk=True)
        
        assert len(result) == 1
        mock_save_chunks.assert_called_once()
    
    @patch('src.embedding.document_processor.DocumentLoader')
    def test_process_documents_no_save(self, mock_loader_class, sample_document):
        """Test processing without saving to disk."""
        mock_loader = Mock()
        mock_loader.load_all_documents.return_value = [sample_document]
        mock_loader_class.return_value = mock_loader
        
        processor = DocumentProcessor(chunk_strategy=CHUNK_STRATEGY_NONE)
        result = processor.process_documents(save_to_disk=False)
        
        assert len(result) == 1


class TestProcessArxivAbstracts:
    """Test process_arxiv_abstracts convenience function."""
    
    @patch('src.embedding.document_processor.DocumentProcessor')
    def test_process_arxiv_abstracts(self, mock_processor_class):
        """Test process_arxiv_abstracts function."""
        mock_processor = Mock()
        mock_processor.process_documents.return_value = [Mock()]
        mock_processor_class.return_value = mock_processor
        
        result = process_arxiv_abstracts(max_documents=10)
        
        assert len(result) == 1
        mock_processor.process_documents.assert_called_once_with(
            include_full_text=False,
            max_documents=10
        )
        # Verify chunk strategy is NONE
        mock_processor_class.assert_called_once()
        call_kwargs = mock_processor_class.call_args[1]
        assert call_kwargs["chunk_strategy"] == CHUNK_STRATEGY_NONE


