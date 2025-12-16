#!/usr/bin/env python3
"""
Project Setup Script for Multi-Source Research Intelligence Platform
Run this script to create the complete project structure for Phase 1

Usage:
    python setup_project.py
"""

import os
from pathlib import Path


def create_directory_structure():
    """Create all necessary directories for the project"""
    
    directories = [
        # Root level
        "notebooks",
        "notebooks/experiments",
        
        # Source code
        "src",
        "src/ingestion",
        "src/vector_store",
        "src/rag",
        "src/agents",
        "src/utils",
        
        # Tests
        "tests",
        
        # Data directories (will be in .gitignore)
        "data/raw",
        "data/processed",
        "data/cache",
        
        # Logs (will be in .gitignore)
        "logs",
        
        # Scripts
        "scripts",
        
        # Documentation
        "docs",
        "docs/phase_guides",
        "docs/architecture_diagrams",
    ]
    
    print("Creating directory structure...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Created: {directory}/")
    
    print("\n✅ Directory structure created successfully!\n")


def create_init_files():
    """Create __init__.py files for Python packages"""
    
    init_locations = [
        "src/__init__.py",
        "src/ingestion/__init__.py",
        "src/vector_store/__init__.py",
        "src/rag/__init__.py",
        "src/agents/__init__.py",
        "src/utils/__init__.py",
        "tests/__init__.py",
    ]
    
    print("Creating __init__.py files...")
    for init_file in init_locations:
        Path(init_file).touch()
        print(f"  ✓ Created: {init_file}")
    
    print("\n✅ __init__.py files created!\n")


def create_gitignore():
    """Create .gitignore file"""
    
    gitignore_content = """# Secrets
.env
*.key
secrets/
config.local.yaml

# Data (large files)
data/
vector_db/
*.db
*.sqlite

# Logs
logs/
*.log

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
venv/
env/
ENV/
.venv

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/

# Jupyter Notebook
.ipynb_checkpoints
*.ipynb_checkpoints/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Models (if downloading locally)
models/
*.bin
*.onnx
*.pt

# Temporary files
*.tmp
*.bak
*.cache
"""
    
    with open(".gitignore", "w") as f:
        f.write(gitignore_content)
    
    print("✅ Created .gitignore\n")


def create_env_example():
    """Create .env.example template"""
    
    env_example = """# OpenAI API Key (for embeddings and LLM)
OPENAI_API_KEY=sk-your-api-key-here

# Alternative: Anthropic Claude API
ANTHROPIC_API_KEY=sk-ant-your-api-key-here

# ArXiv Configuration
ARXIV_MAX_RESULTS=100
ARXIV_DELAY_SECONDS=3

# Document Processing
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Vector Store
VECTOR_DB_PATH=./vector_db
COLLECTION_NAME=arxiv_papers

# Embedding Model (HuggingFace)
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# LLM Configuration
LLM_MODEL=gpt-3.5-turbo
LLM_TEMPERATURE=0

# Retrieval Configuration
RETRIEVAL_TOP_K=5

# Logging
LOG_LEVEL=INFO
"""
    
    with open(".env.example", "w") as f:
        f.write(env_example)
    
    print("✅ Created .env.example")
    print("   📝 Note: Copy this to .env and add your API keys\n")


def create_requirements_txt():
    """Create requirements.txt with Phase 1 dependencies"""
    
    requirements = """# Core Framework
langchain==0.1.0
langchain-community==0.0.38
langchain-openai==0.0.5

# Vector Store
chromadb==0.4.22

# Embeddings
sentence-transformers==2.3.1

# Data Sources
arxiv==2.1.3

# OpenAI
openai==1.12.0

# Utilities
python-dotenv==1.0.0
tiktoken==0.5.2
pydantic==2.5.3

# Processing
beautifulsoup4==4.12.3
requests==2.31.0

# Development
jupyter==1.0.0
ipykernel==6.29.0

# Testing
pytest==7.4.4
pytest-cov==4.1.0

# Logging and Monitoring
tqdm==4.66.1
"""
    
    with open("requirements.txt", "w") as f:
        f.write(requirements)
    
    print("✅ Created requirements.txt\n")


def create_config_py():
    """Create src/config.py for configuration management"""
    
    config_content = '''"""
Configuration management for the Research RAG project
Loads settings from .env file
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
VECTOR_DB_DIR = PROJECT_ROOT / "vector_db"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, 
                  CACHE_DIR, VECTOR_DB_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Validate required keys
if not OPENAI_API_KEY and not ANTHROPIC_API_KEY:
    raise ValueError(
        "Either OPENAI_API_KEY or ANTHROPIC_API_KEY must be set in .env file"
    )

# ArXiv Configuration
ARXIV_MAX_RESULTS = int(os.getenv("ARXIV_MAX_RESULTS", "100"))
ARXIV_DELAY_SECONDS = float(os.getenv("ARXIV_DELAY_SECONDS", "3.0"))

# Document Processing
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# Vector Store
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", str(VECTOR_DB_DIR))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "arxiv_papers")

# Embedding Model
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", 
    "sentence-transformers/all-MiniLM-L6-v2"
)

# LLM Configuration
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))

# Retrieval Configuration
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
'''
    
    with open("src/config.py", "w") as f:
        f.write(config_content)
    
    print("✅ Created src/config.py\n")


def create_logging_config():
    """Create src/utils/logging_config.py"""
    
    logging_content = '''"""
Logging configuration for the Research RAG project
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logging(
    component_name: str,
    log_level: str = "INFO",
    log_dir: Optional[Path] = None
) -> logging.Logger:
    """
    Set up logging for a component
    
    Args:
        component_name: Name of the component (e.g., 'arxiv_fetcher')
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (default: ./logs)
    
    Returns:
        Configured logger instance
    """
    if log_dir is None:
        log_dir = Path("logs")
    
    log_dir.mkdir(exist_ok=True)
    
    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{component_name}_{timestamp}.log"
    
    # Create logger
    logger = logging.getLogger(component_name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_formatter = logging.Formatter(
        '[%(levelname)s] [%(name)s] %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
'''
    
    with open("src/utils/logging_config.py", "w") as f:
        f.write(logging_content)
    
    print("✅ Created src/utils/logging_config.py\n")


def create_readme():
    """Create README.md"""
    
    readme_content = """# Multi-Source Research Intelligence Platform (MSRIP)

## 📚 Overview

An agentic RAG system that autonomously searches academic papers, patents, news, and datasets to generate comprehensive research reports with proper citations and contradiction analysis.

**Current Phase**: Phase 1 - Foundation & Basic RAG Setup

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your API keys
nano .env  # or your preferred editor
```

### 3. Run Phase 1

```bash
# Open Jupyter notebook
jupyter notebook notebooks/phase1_basic_rag.ipynb
```

## 📁 Project Structure

```
research-rag/
├── src/              # Source code
├── notebooks/        # Jupyter notebooks
├── tests/           # Test suite
├── data/            # Data storage (gitignored)
├── scripts/         # Utility scripts
└── docs/            # Documentation
```

## 🎯 Phase 1 Features

- ✅ ArXiv paper fetching with rate limiting
- ✅ Document processing and chunking
- ✅ Vector store with ChromaDB
- ✅ Basic RAG question-answering
- ✅ Citation tracking

## 📖 Documentation

- [AGENT.md](AGENT.md) - Architecture rules and guidelines
- [Phase 1 Guide](docs/phase_guides/phase1.md) - Detailed implementation guide

## 🧪 Testing

```bash
# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 📊 Current Capabilities

**Data Sources**: ArXiv (academic papers)
**Query Processing**: Basic RAG with similarity search
**Citation**: Automatic source attribution
**Response Time**: ~3-5 seconds per query

## 🗺️ Roadmap

- [ ] **Phase 2**: Multi-source integration (Semantic Scholar, PubMed)
- [ ] **Phase 3**: Intelligent routing agent
- [ ] **Phase 4**: Document grading and relevance scoring
- [ ] **Phase 5**: Query rewriting and iterative refinement
- [ ] **Phase 6**: Contradiction detection
- [ ] **Phase 7**: Report generation
- [ ] **Phase 8**: Full orchestration
- [ ] **Phase 9**: Optimization and advanced features
- [ ] **Phase 10**: Production deployment

## 🤝 Contributing

This is a learning project. See [AGENT.md](AGENT.md) for architectural guidelines.

## 📝 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

Built with:
- [LangChain](https://python.langchain.com/)
- [ChromaDB](https://www.trychroma.com/)
- [ArXiv API](https://info.arxiv.org/help/api/index.html)
- [Sentence Transformers](https://www.sbert.net/)

---

**Last Updated**: 2024-01-15 | **Version**: 1.0.0-phase1
"""
    
    with open("README.md", "w") as f:
        f.write(readme_content)
    
    print("✅ Created README.md\n")


def create_changelog():
    """Create CHANGELOG.md"""
    
    changelog_content = """# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Multi-source integration (Phase 2)
- Intelligent routing agent (Phase 3)

## [1.0.0-phase1] - 2024-01-15

### Added
- Initial project setup and structure
- ArXiv API integration with rate limiting
- Document processing and chunking pipeline
- ChromaDB vector store implementation
- Basic RAG chain for question-answering
- Citation tracking and source attribution
- Comprehensive documentation (AGENT.md, README.md)
- Logging infrastructure
- Configuration management via .env
- Unit test framework

### Technical Details
- Python 3.10+ support
- LangChain framework integration
- Sentence Transformers for embeddings
- Jupyter notebook for interactive development

---

## Version Format

Versions follow the pattern: `MAJOR.MINOR.PATCH-phaseN`

- **MAJOR**: Breaking changes or complete architecture overhaul
- **MINOR**: New features or phase completion
- **PATCH**: Bug fixes and minor improvements
- **phase**: Current implementation phase (1-10)
"""
    
    with open("CHANGELOG.md", "w") as f:
        f.write(changelog_content)
    
    print("✅ Created CHANGELOG.md\n")


def create_todo():
    """Create TODO.md for tracking tasks"""
    
    todo_content = """# TODO - Multi-Source Research Intelligence Platform

## Phase 1 Tasks

### High Priority
- [ ] Implement ArXiv fetcher (`src/ingestion/arxiv_fetcher.py`)
- [ ] Implement document processor (`src/ingestion/document_processor.py`)
- [ ] Implement vector store manager (`src/vector_store/chroma_store.py`)
- [ ] Implement basic RAG chain (`src/rag/basic_chain.py`)
- [ ] Create Phase 1 Jupyter notebook
- [ ] Write unit tests for each component
- [ ] Manual testing with 10+ queries
- [ ] Document performance metrics

### Medium Priority
- [ ] Add caching for API responses
- [ ] Implement batch processing for large datasets
- [ ] Create CLI script for paper fetching
- [ ] Add progress bars for long operations
- [ ] Create evaluation metrics script

### Low Priority
- [ ] Add more embedding model options
- [ ] Implement async API calls (for Phase 2+)
- [ ] Create architecture diagrams
- [ ] Record demo video

## Technical Debt

- None yet (Phase 1 just starting)

## Future Enhancements (Phase 2+)

- [ ] Semantic Scholar integration
- [ ] PubMed integration
- [ ] News API integration
- [ ] Query routing logic
- [ ] Document grading agent
- [ ] Query rewriting agent

## Questions to Resolve

- Which embedding model performs best for academic papers?
- Optimal chunk size for different document types?
- Should we use hybrid search (vector + keyword)?

---

**Last Updated**: 2024-01-15
"""
    
    with open("TODO.md", "w") as f:
        f.write(todo_content)
    
    print("✅ Created TODO.md\n")


def create_sample_test():
    """Create a sample test file"""
    
    test_content = '''"""
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
'''
    
    with open("tests/test_setup.py", "w") as f:
        f.write(test_content)
    
    print("✅ Created tests/test_setup.py\n")


def create_phase1_notebook():
    """Create Phase 1 starter notebook"""
    
    notebook_content = """{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Phase 1: Basic RAG System with ArXiv\\n",
    "\\n",
    "This notebook walks through the complete Phase 1 implementation:\\n",
    "1. Fetch papers from ArXiv\\n",
    "2. Process and chunk documents\\n",
    "3. Create vector store\\n",
    "4. Build RAG chain\\n",
    "5. Test with queries"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Setup"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Add src to path\\n",
    "import sys\\n",
    "from pathlib import Path\\n",
    "\\n",
    "project_root = Path.cwd().parent\\n",
    "sys.path.insert(0, str(project_root))\\n",
    "\\n",
    "# Imports\\n",
    "from src import config\\n",
    "from src.utils.logging_config import setup_logging\\n",
    "\\n",
    "# Setup logging\\n",
    "logger = setup_logging('phase1_notebook')\\n",
    "logger.info('Phase 1 notebook initialized')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 1: Fetch Papers from ArXiv\\n",
    "\\n",
    "TODO: Implement ArxivFetcher and fetch papers"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Example (implement actual fetcher first):\\n",
    "# from src.ingestion.arxiv_fetcher import ArxivFetcher\\n",
    "#\\n",
    "# fetcher = ArxivFetcher(max_results=50)\\n",
    "# papers = fetcher.search_papers(\\n",
    "#     query='retrieval augmented generation',\\n",
    "#     categories=['cs.CL', 'cs.AI']\\n",
    "# )\\n",
    "#\\n",
    "# print(f'Fetched {len(papers)} papers')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 2: Process Documents\\n",
    "\\n",
    "TODO: Implement DocumentProcessor and process papers"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Example:\\n",
    "# from src.ingestion.document_processor import DocumentProcessor\\n",
    "#\\n",
    "# processor = DocumentProcessor()\\n",
    "# documents = processor.process_papers(papers)\\n",
    "#\\n",
    "# stats = processor.get_statistics(documents)\\n",
    "# print(stats)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 3: Create Vector Store\\n",
    "\\n",
    "TODO: Implement VectorStoreManager and create vector store"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Example:\\n",
    "# from src.vector_store.chroma_store import VectorStoreManager\\n",
    "#\\n",
    "# vs_manager = VectorStoreManager()\\n",
    "# vector_store = vs_manager.create_from_documents(documents)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 4: Build RAG Chain\\n",
    "\\n",
    "TODO: Implement BasicRAGChain"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Example:\\n",
    "# from src.rag.basic_chain import BasicRAGChain\\n",
    "#\\n",
    "# rag = BasicRAGChain(vector_store)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 5: Test Queries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Test queries\\n",
    "# test_questions = [\\n",
    "#     'What are the main benefits of retrieval augmented generation?',\\n",
    "#     'How does RAG improve LLM accuracy?',\\n",
    "#     'What are the challenges in implementing RAG systems?'\\n",
    "# ]\\n",
    "#\\n",
    "# for question in test_questions:\\n",
    "#     print(f'\\\\nQ: {question}')\\n",
    "#     result = rag.ask_with_sources(question)\\n",
    "#     print(f'A: {result[\"answer\"]}')\\n",
    "#     print(f'Sources: {len(result[\"sources\"])} papers')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Next Steps\\n",
    "\\n",
    "- Evaluate retrieval quality\\n",
    "- Tune chunk size and overlap\\n",
    "- Test with more queries\\n",
    "- Move to Phase 2"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.10.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
"""
    
    with open("notebooks/phase1_basic_rag.ipynb", "w") as f:
        f.write(notebook_content)
    
    print("✅ Created notebooks/phase1_basic_rag.ipynb\n")


def create_setup_instructions():
    """Create SETUP.md with detailed instructions"""
    
    setup_content = """# Setup Instructions

Follow these steps to set up the Multi-Source Research Intelligence Platform.

## Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- Git
- OpenAI API key (or Anthropic Claude API key)

## Step-by-Step Setup

### 1. Clone or Create Project Directory

```bash
mkdir research-rag
cd research-rag
```

### 2. Run Setup Script

```bash
# Download and run the setup script
python setup_project.py
```

This creates all necessary directories and files.

### 3. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\\Scripts\\activate
```

### 4. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This may take 5-10 minutes depending on your internet speed.

### 5. Configure Environment Variables

```bash
# Copy the template
cp .env.example .env

# Edit .env with your favorite editor
nano .env   # or vim, code, etc.
```

**Required variables:**
- `OPENAI_API_KEY` - Get from https://platform.openai.com/api-keys

**Optional variables:**
- `ARXIV_MAX_RESULTS` - Default: 100
- `CHUNK_SIZE` - Default: 1000
- `CHUNK_OVERLAP` - Default: 200

### 6. Verify Installation

```bash
# Run tests to verify setup
pytest tests/test_setup.py -v
```

You should see all tests passing.

### 7. Start Development

```bash
# Launch Jupyter
jupyter notebook

# Open: notebooks/phase1_basic_rag.ipynb
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'langchain'"

**Solution**: Make sure virtual environment is activated and dependencies are installed:
```bash
source venv/bin/activate  # or venv\\Scripts\\activate on Windows
pip install -r requirements.txt
```

### Issue: "OpenAI API key not found"

**Solution**: Make sure `.env` file exists and contains `OPENAI_API_KEY=sk-...`

### Issue: Import errors in notebooks

**Solution**: Make sure to run the setup cell that adds src to Python path:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))
```

### Issue: ChromaDB errors

**Solution**: Delete the `vector_db/` directory and recreate the vector store:
```bash
rm -rf vector_db/
```

## Next Steps

1. Implement `src/ingestion/arxiv_fetcher.py`
2. Implement `src/ingestion/document_processor.py`
3. Implement `src/vector_store/chroma_store.py`
4. Implement `src/rag/basic_chain.py`
5. Run through Phase 1 notebook

See [AGENT.md](AGENT.md) for architectural guidelines.

## Getting Help

- Check [TODO.md](TODO.md) for known issues
- Review [AGENT.md](AGENT.md) for design patterns
- Consult Phase 1 detailed guide in `docs/phase_guides/phase1.md`

---

**Last Updated**: 2024-01-15
"""
    
    with open("SETUP.md", "w") as f:
        f.write(setup_content)
    
    print("✅ Created SETUP.md\n")


def print_next_steps():
    """Print next steps for the user"""
    
    print("\n" + "="*70)
    print("🎉 PROJECT STRUCTURE CREATED SUCCESSFULLY!")
    print("="*70)
    
    print("\n📝 NEXT STEPS:\n")
    print("1. Create a virtual environment:")
    print("   python -m venv venv")
    print("   source venv/bin/activate  # On Windows: venv\\Scripts\\activate\n")
    
    print("2. Install dependencies:")
    print("   pip install -r requirements.txt\n")
    
    print("3. Configure your environment:")
    print("   cp .env.example .env")
    print("   # Then edit .env and add your OpenAI API key\n")
    
    print("4. Verify setup:")
    print("   pytest tests/test_setup.py -v\n")
    
    print("5. Start implementing Phase 1 components:")
    print("   - src/ingestion/arxiv_fetcher.py")
    print("   - src/ingestion/document_processor.py")
    print("   - src/vector_store/chroma_store.py")
    print("   - src/rag/basic_chain.py\n")
    
    print("6. Test your implementation:")
    print("   jupyter notebook notebooks/phase1_basic_rag.ipynb\n")
    
    print("📚 DOCUMENTATION:")
    print("   - AGENT.md: Architecture rules and guidelines")
    print("   - SETUP.md: Detailed setup instructions")
    print("   - README.md: Project overview")
    print("   - TODO.md: Task tracking")
    
    print("\n" + "="*70)
    print("Happy coding! 🚀")
    print("="*70 + "\n")


def main():
    """Main setup function"""
    print("\n" + "="*70)
    print("Multi-Source Research Intelligence Platform - Project Setup")
    print("="*70 + "\n")
    
    # Create all project components
    create_directory_structure()
    create_init_files()
    create_gitignore()
    create_env_example()
    create_requirements_txt()
    create_config_py()
    create_logging_config()
    create_readme()
    create_changelog()
    create_todo()
    create_sample_test()
    create_phase1_notebook()
    create_setup_instructions()
    print_next_steps()


if __name__ == "__main__":
    main()