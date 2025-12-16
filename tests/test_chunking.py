"""
Tests for document chunking module.
"""

import pytest
from langchain_classic.schema import Document

from src.embedding.chunking import DocumentChunker
from src.constants import (
    ChunkingStrategy,
    CHUNK_STRATEGY_NONE,
    CHUNK_STRATEGY_RECURSIVE,
    CHUNK_STRATEGY_TOKEN
)


class TestDocumentChunker:
    """Test DocumentChunker class."""
    
    def test_init_default(self):
        """Test DocumentChunker initialization with defaults."""
        chunker = DocumentChunker()
        assert chunker.chunk_size == 512
        assert chunker.chunk_overlap == 50
        assert chunker.strategy == CHUNK_STRATEGY_RECURSIVE
    
    def test_init_with_enum(self):
        """Test initialization with ChunkingStrategy enum."""
        chunker = DocumentChunker(strategy=ChunkingStrategy.NONE)
        assert chunker.strategy == CHUNK_STRATEGY_NONE
    
    def test_init_with_string(self):
        """Test initialization with string constant."""
        chunker = DocumentChunker(strategy=CHUNK_STRATEGY_TOKEN)
        assert chunker.strategy == CHUNK_STRATEGY_TOKEN
    
    def test_chunk_document_none_strategy(self, sample_document):
        """Test chunking with 'none' strategy (no chunking)."""
        chunker = DocumentChunker(strategy=CHUNK_STRATEGY_NONE)
        chunks = chunker.chunk_document(sample_document)
        
        assert len(chunks) == 1
        assert chunks[0].page_content == sample_document.page_content
        assert chunks[0].metadata["chunk_index"] == 0
        assert chunks[0].metadata["total_chunks"] == 1
        assert "chunk_id" in chunks[0].metadata
    
    def test_chunk_document_recursive_strategy(self, sample_document):
        """Test chunking with 'recursive' strategy."""
        chunker = DocumentChunker(
            strategy=CHUNK_STRATEGY_RECURSIVE,
            chunk_size=100,
            chunk_overlap=10
        )
        chunks = chunker.chunk_document(sample_document)
        
        assert len(chunks) >= 1
        assert all(isinstance(chunk, Document) for chunk in chunks)
        assert all("chunk_id" in chunk.metadata for chunk in chunks)
        assert all("chunk_index" in chunk.metadata for chunk in chunks)
    
    def test_chunk_document_invalid_strategy(self):
        """Test chunking with invalid strategy."""
        chunker = DocumentChunker(strategy="invalid")
        
        with pytest.raises(ValueError, match="Unknown strategy"):
            chunker._init_splitter()
    
    def test_add_chunk_metadata(self, sample_document):
        """Test adding chunk metadata."""
        chunker = DocumentChunker()
        enriched = chunker._add_chunk_metadata(
            sample_document,
            chunk_index=0,
            total_chunks=5
        )
        
        assert enriched.metadata["chunk_index"] == 0
        assert enriched.metadata["total_chunks"] == 5
        assert enriched.metadata["chunk_id"] is not None
        assert enriched.metadata["chunk_size"] == len(enriched.page_content)
        # Original metadata should be preserved
        assert enriched.metadata["arxiv_id"] == sample_document.metadata["arxiv_id"]
    
    def test_chunk_documents_multiple(self, sample_documents):
        """Test chunking multiple documents."""
        chunker = DocumentChunker(strategy=CHUNK_STRATEGY_NONE)
        all_chunks = chunker.chunk_documents(sample_documents)
        
        assert len(all_chunks) == len(sample_documents)
        assert all(isinstance(chunk, Document) for chunk in all_chunks)
    
    def test_chunk_preserves_metadata(self, sample_document):
        """Test that chunking preserves original document metadata."""
        chunker = DocumentChunker(strategy=CHUNK_STRATEGY_NONE)
        chunks = chunker.chunk_document(sample_document)
        
        chunk = chunks[0]
        original_metadata = sample_document.metadata
        
        # All original metadata should be present
        for key, value in original_metadata.items():
            assert chunk.metadata[key] == value
        
        # Plus chunk-specific metadata
        assert "chunk_id" in chunk.metadata
        assert "chunk_index" in chunk.metadata

