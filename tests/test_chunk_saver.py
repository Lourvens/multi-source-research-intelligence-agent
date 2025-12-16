"""
Tests for chunk saver module.
"""

import pytest
import json
import numpy as np
from pathlib import Path
from langchain_classic.schema import Document

from src.embedding.chunk_saver import save_processed_chunks, load_processed_chunks
from src.constants import DataSource, DATA_SOURCE_ARXIV


class TestSaveProcessedChunks:
    """Test save_processed_chunks function."""
    
    def test_save_chunks_success(self, temp_data_dir, sample_document, monkeypatch):
        """Test saving chunks to disk."""
        from src import config
        original_processed_dir = config.PROCESSED_DATA_DIR
        monkeypatch.setattr(config, "PROCESSED_DATA_DIR", temp_data_dir["data"] / "processed")
        
        # Add embedding to document
        sample_document.metadata["embedding"] = np.array([0.1, 0.2, 0.3, 0.4])
        
        try:
            saved_path = save_processed_chunks(
                [sample_document],
                source=DATA_SOURCE_ARXIV,
                suffix="abstracts"
            )
            
            assert saved_path is not None
            assert saved_path.exists()
            
            # Verify file content
            with saved_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            
            assert data["source"] == DATA_SOURCE_ARXIV
            assert data["total_chunks"] == 1
            assert len(data["chunks"]) == 1
            assert data["chunks"][0]["page_content"] == sample_document.page_content
            # Embedding should be converted to list
            assert isinstance(data["chunks"][0]["metadata"]["embedding"], list)
        finally:
            monkeypatch.setattr(config, "PROCESSED_DATA_DIR", original_processed_dir)
    
    def test_save_chunks_empty_list(self, temp_data_dir, monkeypatch):
        """Test saving empty chunk list."""
        from src import config
        original_processed_dir = config.PROCESSED_DATA_DIR
        monkeypatch.setattr(config, "PROCESSED_DATA_DIR", temp_data_dir["data"] / "processed")
        
        try:
            saved_path = save_processed_chunks([], source=DATA_SOURCE_ARXIV)
            assert saved_path is None
        finally:
            monkeypatch.setattr(config, "PROCESSED_DATA_DIR", original_processed_dir)
    
    def test_save_chunks_with_enum(self, temp_data_dir, sample_document, monkeypatch):
        """Test saving with DataSource enum."""
        from src import config
        original_processed_dir = config.PROCESSED_DATA_DIR
        monkeypatch.setattr(config, "PROCESSED_DATA_DIR", temp_data_dir["data"] / "processed")
        
        sample_document.metadata["embedding"] = np.array([0.1, 0.2, 0.3])
        
        try:
            saved_path = save_processed_chunks(
                [sample_document],
                source=DataSource.ARXIV,
                suffix="test"
            )
            
            assert saved_path is not None
            # Verify source is correct
            with saved_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            assert data["source"] == DATA_SOURCE_ARXIV
        finally:
            monkeypatch.setattr(config, "PROCESSED_DATA_DIR", original_processed_dir)
    
    def test_save_chunks_preserves_metadata(self, temp_data_dir, sample_document, monkeypatch):
        """Test that saving preserves all metadata."""
        from src import config
        original_processed_dir = config.PROCESSED_DATA_DIR
        monkeypatch.setattr(config, "PROCESSED_DATA_DIR", temp_data_dir["data"] / "processed")
        
        sample_document.metadata["embedding"] = np.array([0.1, 0.2, 0.3])
        sample_document.metadata["custom_field"] = "test_value"
        
        try:
            saved_path = save_processed_chunks([sample_document], source=DATA_SOURCE_ARXIV)
            
            with saved_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            
            saved_metadata = data["chunks"][0]["metadata"]
            assert saved_metadata["custom_field"] == "test_value"
            assert saved_metadata["arxiv_id"] == sample_document.metadata["arxiv_id"]
        finally:
            monkeypatch.setattr(config, "PROCESSED_DATA_DIR", original_processed_dir)


class TestLoadProcessedChunks:
    """Test load_processed_chunks function."""
    
    def test_load_chunks_success(self, temp_data_dir, sample_document, monkeypatch):
        """Test loading chunks from disk."""
        from src import config
        original_processed_dir = config.PROCESSED_DATA_DIR
        monkeypatch.setattr(config, "PROCESSED_DATA_DIR", temp_data_dir["data"] / "processed")
        
        # First save chunks
        sample_document.metadata["embedding"] = np.array([0.1, 0.2, 0.3, 0.4])
        saved_path = save_processed_chunks([sample_document], source=DATA_SOURCE_ARXIV)
        
        try:
            # Then load them
            loaded_chunks = load_processed_chunks(saved_path)
            
            assert len(loaded_chunks) == 1
            assert loaded_chunks[0].page_content == sample_document.page_content
            # Embedding should be restored as numpy array
            assert isinstance(loaded_chunks[0].metadata["embedding"], np.ndarray)
            np.testing.assert_array_equal(
                loaded_chunks[0].metadata["embedding"],
                sample_document.metadata["embedding"]
            )
        finally:
            monkeypatch.setattr(config, "PROCESSED_DATA_DIR", original_processed_dir)
    
    def test_load_chunks_nonexistent_file(self, temp_data_dir):
        """Test loading from non-existent file."""
        nonexistent_path = temp_data_dir["processed"] / "nonexistent.json"
        
        with pytest.raises(Exception):
            load_processed_chunks(nonexistent_path)



