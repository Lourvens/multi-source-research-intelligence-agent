"""
Tests for document embedder module.
"""

import pytest
import numpy as np
from langchain_classic.schema import Document

from src.embedding.embedder import DocumentEmbedder


class TestDocumentEmbedder:
    """Test DocumentEmbedder class."""
    
    def test_init_default(self):
        """Test DocumentEmbedder initialization with default model."""
        embedder = DocumentEmbedder()
        assert embedder.model_name is not None
        assert embedder.embeddings is not None
    
    def test_init_custom_model(self):
        """Test initialization with custom model name."""
        embedder = DocumentEmbedder(model_name="all-MiniLM-L6-v2")
        assert "all-MiniLM-L6-v2" in embedder.model_name
    
    def test_init_auto_prefix(self):
        """Test that model name gets sentence-transformers prefix if missing."""
        embedder = DocumentEmbedder(model_name="all-MiniLM-L6-v2")
        assert embedder.model_name.startswith("sentence-transformers/")
    
    def test_embed_documents_single(self, sample_document):
        """Test embedding a single document."""
        embedder = DocumentEmbedder()
        embedded = embedder.embed_documents([sample_document])
        
        assert len(embedded) == 1
        assert "embedding" in embedded[0].metadata
        # Embeddings are returned as lists, convert to numpy for testing
        embedding = embedded[0].metadata["embedding"]
        assert isinstance(embedding, (list, np.ndarray))
        if isinstance(embedding, list):
            embedding = np.array(embedding)
        assert len(embedding) > 0
    
    def test_embed_documents_multiple(self, sample_documents):
        """Test embedding multiple documents."""
        embedder = DocumentEmbedder()
        embedded = embedder.embed_documents(sample_documents)
        
        assert len(embedded) == len(sample_documents)
        assert all("embedding" in doc.metadata for doc in embedded)
        # Embeddings can be lists or numpy arrays
        for doc in embedded:
            embedding = doc.metadata["embedding"]
            assert isinstance(embedding, (list, np.ndarray))
    
    def test_embed_documents_preserves_metadata(self, sample_document):
        """Test that embedding preserves document metadata."""
        embedder = DocumentEmbedder()
        embedded = embedder.embed_documents([sample_document])
        
        embedded_doc = embedded[0]
        original_metadata = sample_document.metadata.copy()
        
        # All original metadata should be preserved
        for key, value in original_metadata.items():
            assert embedded_doc.metadata[key] == value
        
        # Plus embedding
        assert "embedding" in embedded_doc.metadata
    
    def test_embed_documents_batch_size(self, sample_documents):
        """Test embedding with different batch sizes."""
        embedder = DocumentEmbedder()
        
        # Test with small batch size
        embedded = embedder.embed_documents(sample_documents, batch_size=1)
        assert len(embedded) == len(sample_documents)
    
    def test_embed_documents_empty_list(self):
        """Test embedding empty document list."""
        embedder = DocumentEmbedder()
        embedded = embedder.embed_documents([])
        assert len(embedded) == 0
    
    def test_embed_documents_embedding_dimension(self, sample_document):
        """Test that embeddings have consistent dimensions."""
        embedder = DocumentEmbedder()
        embedded = embedder.embed_documents([sample_document])
        
        embedding = embedded[0].metadata["embedding"]
        # Convert to numpy array if it's a list
        if isinstance(embedding, list):
            embedding = np.array(embedding)
        assert embedding.ndim == 1  # Should be 1D array
        assert embedding.shape[0] == 384  # all-MiniLM-L6-v2 produces 384-dim embeddings


