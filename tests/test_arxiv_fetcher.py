"""
Tests for ArXiv fetcher module.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.ingestion.arxiv_fetcher import (
    ArxivFetcher,
    batch_fetch_arxiv_metadata,
    save_metadatas
)
from src.constants import DATA_SOURCE_ARXIV


class TestArxivFetcher:
    """Test ArxivFetcher function."""
    
    @patch('src.ingestion.arxiv_fetcher.arxiv.Client')
    def test_arxiv_fetcher_success(self, mock_client_class, sample_arxiv_metadata):
        """Test successful ArXiv fetching."""
        # Mock ArXiv client and results
        mock_client = Mock()
        mock_result = Mock()
        mock_result.get_short_id.return_value = sample_arxiv_metadata["id"]
        mock_result.title = sample_arxiv_metadata["title"]
        mock_result.summary = sample_arxiv_metadata["summary"]
        mock_result.authors = [Mock(name=name) for name in sample_arxiv_metadata["authors"]]
        mock_result.published.isoformat.return_value = sample_arxiv_metadata["published"]
        mock_result.updated.isoformat.return_value = sample_arxiv_metadata["updated"]
        mock_result.links = [Mock(href="https://arxiv.org/abs/1808.01591v1")]
        mock_result.categories = sample_arxiv_metadata["categories"]
        mock_result.pdf_url = sample_arxiv_metadata["pdf_url"]
        
        mock_search = Mock()
        mock_client.results.return_value = [mock_result]
        mock_client_class.return_value = mock_client
        
        # Call function
        result = ArxivFetcher(max_results=1)
        
        # Assertions
        assert len(result) == 1
        assert result[0]["id"] == sample_arxiv_metadata["id"]
        assert result[0]["title"] == sample_arxiv_metadata["title"]
        assert result[0]["source"] == DATA_SOURCE_ARXIV
        assert "fetched_at" in result[0]
    
    @patch('src.ingestion.arxiv_fetcher.arxiv.Client')
    def test_arxiv_fetcher_empty_result(self, mock_client_class):
        """Test ArXiv fetcher with empty results."""
        mock_client = Mock()
        mock_client.results.return_value = []
        mock_client_class.return_value = mock_client
        
        result = ArxivFetcher(max_results=10)
        assert len(result) == 0
    
    @patch('src.ingestion.arxiv_fetcher.arxiv.Client')
    def test_arxiv_fetcher_error_handling(self, mock_client_class):
        """Test error handling in ArXiv fetcher."""
        mock_client = Mock()
        mock_client.results.side_effect = Exception("API Error")
        mock_client_class.return_value = mock_client
        
        with pytest.raises(Exception):
            ArxivFetcher(max_results=10)


class TestBatchFetchArxivMetadata:
    """Test batch_fetch_arxiv_metadata function."""
    
    @patch('src.ingestion.arxiv_fetcher.ArxivFetcher')
    @patch('src.ingestion.arxiv_fetcher.time.sleep')
    def test_batch_fetch_single_batch(self, mock_sleep, mock_fetcher):
        """Test batch fetching with single batch."""
        mock_fetcher.return_value = [{"id": "1808.01591v1", "title": "Test"}]
        
        result = batch_fetch_arxiv_metadata(
            num_batches=1,
            results_per_batch=10,
            delay_seconds=0.1
        )
        
        assert len(result) == 1
        assert result[0]["id"] == "1808.01591v1"
        mock_sleep.assert_not_called()  # No delay for first batch
    
    @patch('src.ingestion.arxiv_fetcher.ArxivFetcher')
    @patch('src.ingestion.arxiv_fetcher.time.sleep')
    def test_batch_fetch_multiple_batches(self, mock_sleep, mock_fetcher):
        """Test batch fetching with multiple batches."""
        mock_fetcher.return_value = [{"id": f"1808.0159{i}v1", "title": f"Test {i}"} for i in range(5)]
        
        result = batch_fetch_arxiv_metadata(
            num_batches=3,
            results_per_batch=5,
            delay_seconds=0.1
        )
        
        assert len(result) == 15  # 3 batches * 5 results
        assert mock_fetcher.call_count == 3
        assert mock_sleep.call_count == 2  # Delay between batches (not before first)
    
    @patch('src.ingestion.arxiv_fetcher.ArxivFetcher')
    def test_batch_fetch_max_results(self, mock_fetcher):
        """Test batch fetching with max_results limit."""
        mock_fetcher.return_value = [{"id": f"1808.0159{i}v1"} for i in range(10)]
        
        result = batch_fetch_arxiv_metadata(
            num_batches=None,
            results_per_batch=10,
            max_results=25
        )
        
        assert len(result) == 25  # Should respect max_results
        assert mock_fetcher.call_count == 3  # ceil(25/10) = 3 batches


class TestSaveMetadatas:
    """Test save_metadatas function."""
    
    def test_save_metadatas_success(self, temp_data_dir, sample_arxiv_metadata, monkeypatch):
        """Test saving metadata to disk."""
        # Patch config to use temp directory
        from src import config
        original_raw_dir = config.RAW_DATA_DIR
        monkeypatch.setattr(config, "RAW_DATA_DIR", temp_data_dir["data"] / "raw")
        
        try:
            save_metadatas([sample_arxiv_metadata])
            
            # Verify file was created
            metadata_file = temp_data_dir["raw"] / "arxiv_metadata" / f"{sample_arxiv_metadata['id']}.json"
            assert metadata_file.exists()
            
            # Verify content
            with metadata_file.open("r", encoding="utf-8") as f:
                saved_data = json.load(f)
            
            assert saved_data["id"] == sample_arxiv_metadata["id"]
            assert saved_data["title"] == sample_arxiv_metadata["title"]
        finally:
            monkeypatch.setattr(config, "RAW_DATA_DIR", original_raw_dir)
    
    def test_save_metadatas_invalid_data(self, temp_data_dir, monkeypatch):
        """Test saving invalid metadata (should handle gracefully)."""
        from src import config
        original_raw_dir = config.RAW_DATA_DIR
        monkeypatch.setattr(config, "RAW_DATA_DIR", temp_data_dir["data"] / "raw")
        
        invalid_metadata = {"id": None, "title": "Test"}  # Missing required fields
        
        try:
            # Should not raise, but log error
            save_metadatas([invalid_metadata])
        finally:
            monkeypatch.setattr(config, "RAW_DATA_DIR", original_raw_dir)

