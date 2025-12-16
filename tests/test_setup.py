"""
Sample test file to verify pytest setup
"""

import pytest
from pathlib import Path


def test_project_structure():
    """Test that essential directories exist"""
    assert Path("src").exists()
    assert Path("notebooks").exists()
    assert Path("data").exists()
    assert Path("tests").exists()


def test_config_loads():
    """Test that configuration loads without errors"""
    try:
        from src import config
        assert config.CHUNK_SIZE > 0
        assert config.CHUNK_OVERLAP >= 0
    except ImportError:
        pytest.skip("Config not yet implemented")


def test_imports():
    """Test that core dependencies are installed"""
    import langchain
    import chromadb
    import arxiv
    import sentence_transformers
    assert True  # If we got here, imports worked
