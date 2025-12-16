"""
Pytest configuration and shared fixtures for Phase 1 tests.
"""

import pytest
import json
import tempfile
from pathlib import Path
from typing import Dict, List

from langchain_classic.schema import Document


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary data directory structure."""
    data_dir = tmp_path / "data"
    raw_dir = data_dir / "raw" / "arxiv_metadata"
    processed_dir = data_dir / "processed" / "arxiv"
    cache_dir = data_dir / "cache"
    
    for dir_path in [raw_dir, processed_dir, cache_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return {
        "data": data_dir,
        "raw": raw_dir,
        "processed": processed_dir,
        "cache": cache_dir
    }


@pytest.fixture
def sample_arxiv_metadata() -> Dict:
    """Sample ArXiv metadata for testing."""
    return {
        "id": "1808.01591v1",
        "title": "LISA: Explaining Recurrent Neural Network Judgments",
        "summary": "Recurrent neural networks (RNNs) are temporal networks...",
        "authors": ["Pankaj Gupta", "Hinrich Schütze"],
        "published": "2018-08-05T00:00:00Z",
        "updated": "2018-08-05T00:00:00Z",
        "categories": ["cs.CL", "cs.AI"],
        "pdf_url": "https://arxiv.org/pdf/1808.01591v1.pdf",
        "fetched_at": "2024-01-15T10:00:00Z",
        "source": "arxiv"
    }


@pytest.fixture
def sample_arxiv_metadata_file(temp_data_dir, sample_arxiv_metadata):
    """Create a sample metadata JSON file."""
    metadata_file = temp_data_dir["raw"] / f"{sample_arxiv_metadata['id']}.json"
    with metadata_file.open("w", encoding="utf-8") as f:
        json.dump(sample_arxiv_metadata, f, indent=2)
    return metadata_file


@pytest.fixture
def sample_document(sample_arxiv_metadata) -> Document:
    """Create a sample LangChain Document for testing."""
    content = f"""Title: {sample_arxiv_metadata['title']}

Abstract:
{sample_arxiv_metadata['summary']}
"""
    
    return Document(
        page_content=content,
        metadata={
            "arxiv_id": sample_arxiv_metadata["id"],
            "title": sample_arxiv_metadata["title"],
            "authors": sample_arxiv_metadata["authors"],
            "published": sample_arxiv_metadata["published"],
            "source": "arxiv",
            "content_type": "abstract_only"
        }
    )


@pytest.fixture
def sample_documents(sample_document) -> List[Document]:
    """Create multiple sample documents."""
    docs = [sample_document]
    
    # Add a second document
    doc2 = Document(
        page_content="Title: Test Paper 2\n\nAbstract: This is another test paper.",
        metadata={
            "arxiv_id": "2001.00001v1",
            "title": "Test Paper 2",
            "authors": ["Author A", "Author B"],
            "published": "2020-01-01T00:00:00Z",
            "source": "arxiv",
            "content_type": "abstract_only"
        }
    )
    docs.append(doc2)
    
    return docs

