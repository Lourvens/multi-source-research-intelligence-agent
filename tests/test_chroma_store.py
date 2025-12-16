"""
Tests for ChromaDB vector store module.
"""

import pytest
import numpy as np
from pathlib import Path
from langchain_classic.schema import Document

from src.vector_store import ChromaVectorStore
from src.constants import DATA_SOURCE_ARXIV


@pytest.fixture
def temp_vector_db_dir(tmp_path):
    """Create a temporary vector DB directory."""
    vector_db_dir = tmp_path / "vector_db"
    vector_db_dir.mkdir(parents=True, exist_ok=True)
    return vector_db_dir


@pytest.fixture
def sample_documents_with_embeddings():
    """Create sample documents with embeddings."""
    docs = []
    for i in range(3):
        doc = Document(
            page_content=f"This is test document {i} about machine learning and AI.",
            metadata={
                "arxiv_id": f"2001.0000{i}v1",
                "title": f"Test Paper {i}",
                "authors": ["Author A", "Author B"],
                "published": "2020-01-01T00:00:00Z",
                "source": DATA_SOURCE_ARXIV
            }
        )
        docs.append(doc)
    return docs


class TestChromaVectorStore:
    """Test ChromaVectorStore class."""
    
    def test_init_default(self, temp_vector_db_dir):
        """Test initialization with default parameters."""
        store = ChromaVectorStore(persist_directory=temp_vector_db_dir)
        
        assert store.collection_name is not None
        assert store.persist_directory == temp_vector_db_dir
        assert store.embeddings is not None
    
    def test_init_custom_params(self, temp_vector_db_dir):
        """Test initialization with custom parameters."""
        store = ChromaVectorStore(
            collection_name="test_collection",
            persist_directory=temp_vector_db_dir,
            embedding_model="all-MiniLM-L6-v2"
        )
        
        assert store.collection_name == "test_collection"
        assert store.persist_directory == temp_vector_db_dir
    
    def test_create_from_documents(self, temp_vector_db_dir, sample_documents_with_embeddings):
        """Test creating vector store from documents."""
        store = ChromaVectorStore(
            collection_name="test_collection",
            persist_directory=temp_vector_db_dir
        )
        
        vector_store = store.create_from_documents(sample_documents_with_embeddings)
        
        assert vector_store is not None
        assert store.vector_store is not None
        
        # Verify collection exists
        info = store.get_collection_info()
        assert info["exists"] is True
        assert info["count"] == len(sample_documents_with_embeddings)
    
    def test_create_from_empty_documents(self, temp_vector_db_dir):
        """Test creating vector store with empty document list."""
        store = ChromaVectorStore(persist_directory=temp_vector_db_dir)
        
        result = store.create_from_documents([])
        
        assert result is None
    
    def test_similarity_search(self, temp_vector_db_dir, sample_documents_with_embeddings):
        """Test similarity search."""
        store = ChromaVectorStore(
            collection_name="test_collection",
            persist_directory=temp_vector_db_dir
        )
        
        # Create vector store first
        store.create_from_documents(sample_documents_with_embeddings)
        
        # Perform search
        results = store.similarity_search("machine learning", k=2)
        
        assert len(results) == 2
        assert all(isinstance(doc, Document) for doc in results)
        assert all("machine learning" in doc.page_content.lower() or "ai" in doc.page_content.lower() 
                  for doc in results)
    
    def test_similarity_search_with_score(self, temp_vector_db_dir, sample_documents_with_embeddings):
        """Test similarity search with scores."""
        store = ChromaVectorStore(
            collection_name="test_collection",
            persist_directory=temp_vector_db_dir
        )
        
        # Create vector store first
        store.create_from_documents(sample_documents_with_embeddings)
        
        # Perform search with scores
        results = store.similarity_search_with_score("machine learning", k=2)
        
        assert len(results) == 2
        assert all(isinstance(result, tuple) for result in results)
        assert all(isinstance(doc, Document) for doc, score in results)
        assert all(isinstance(score, float) for doc, score in results)
        # ChromaDB returns distance scores (lower is better, so ascending order)
        scores = [score for _, score in results]
        assert scores == sorted(scores)  # Ascending order for distances
    
    def test_add_documents(self, temp_vector_db_dir, sample_documents_with_embeddings):
        """Test adding documents to existing vector store."""
        store = ChromaVectorStore(
            collection_name="test_collection",
            persist_directory=temp_vector_db_dir
        )
        
        # Create initial vector store
        store.create_from_documents(sample_documents_with_embeddings[:2])
        
        # Add more documents
        new_doc = Document(
            page_content="This is a new document about deep learning.",
            metadata={"arxiv_id": "2001.00010v1", "title": "New Paper"}
        )
        
        ids = store.add_documents([new_doc])
        
        assert len(ids) == 1
        
        # Verify collection count increased
        info = store.get_collection_info()
        assert info["count"] == 3
    
    def test_get_collection_info(self, temp_vector_db_dir, sample_documents_with_embeddings):
        """Test getting collection information."""
        # Use unique collection name to avoid conflicts
        collection_name = "test_collection_info"
        store = ChromaVectorStore(
            collection_name=collection_name,
            persist_directory=temp_vector_db_dir
        )
        
        # Before creating, should not exist (don't load it)
        # get_collection_info will try to load, so we check the error case
        info = store.get_collection_info()
        # If collection doesn't exist, it might still return exists=False or error
        # The actual behavior depends on ChromaDB's error handling
        
        # After creating
        store.create_from_documents(sample_documents_with_embeddings)
        info = store.get_collection_info()
        
        assert info["exists"] is True
        assert info["count"] == len(sample_documents_with_embeddings)
        assert info["collection_name"] == collection_name
        assert "embedding_model" in info
    
    def test_persistence(self, temp_vector_db_dir, sample_documents_with_embeddings):
        """Test that vector store persists to disk."""
        # Create and save
        store1 = ChromaVectorStore(
            collection_name="test_collection",
            persist_directory=temp_vector_db_dir
        )
        store1.create_from_documents(sample_documents_with_embeddings)
        
        # Create new instance (should load from disk)
        store2 = ChromaVectorStore(
            collection_name="test_collection",
            persist_directory=temp_vector_db_dir
        )
        
        # Should be able to search without recreating
        results = store2.similarity_search("machine learning", k=1)
        assert len(results) > 0
    
    def test_delete_collection(self, temp_vector_db_dir, sample_documents_with_embeddings):
        """Test deleting a collection."""
        # Use unique collection name
        collection_name = "test_collection_delete"
        store = ChromaVectorStore(
            collection_name=collection_name,
            persist_directory=temp_vector_db_dir
        )
        
        # Create collection
        store.create_from_documents(sample_documents_with_embeddings)
        assert store.get_collection_info()["exists"] is True
        
        # Delete collection
        result = store.delete_collection()
        assert result is True
        
        # Create a new store instance to verify deletion (old instance might cache)
        new_store = ChromaVectorStore(
            collection_name=collection_name,
            persist_directory=temp_vector_db_dir
        )
        info = new_store.get_collection_info()
        # After deletion, collection should not exist
        assert info["exists"] is False


class TestCreateVectorStoreFromProcessedChunks:
    """Test convenience function."""
    
    def test_create_from_processed_chunks(self, temp_vector_db_dir, sample_documents_with_embeddings):
        """Test creating vector store from processed chunks."""
        from src.vector_store import create_vector_store_from_processed_chunks
        
        store = create_vector_store_from_processed_chunks(
            sample_documents_with_embeddings,
            collection_name="test_collection",
            persist_directory=temp_vector_db_dir
        )
        
        assert isinstance(store, ChromaVectorStore)
        assert store.vector_store is not None
        
        info = store.get_collection_info()
        assert info["exists"] is True
        assert info["count"] == len(sample_documents_with_embeddings)

