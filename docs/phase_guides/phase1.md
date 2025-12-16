# Phase 1 Guide: Foundation & Basic RAG Setup

## 📋 Overview

Phase 1 establishes the foundation of the Multi-Source Research Intelligence Platform (MSRIP). This phase implements a complete RAG (Retrieval-Augmented Generation) pipeline for ArXiv academic papers, including data fetching, document processing, chunking, embedding, and vector storage.

**Status**: ✅ Complete  
**Version**: 1.0.0-phase1  
**Date**: December 2024

---

## 🎯 Phase 1 Objectives

### Primary Goals
1. ✅ **ArXiv Integration**: Fetch and store ArXiv paper metadata and PDFs
2. ✅ **Document Processing**: Load, chunk, and embed documents
3. ✅ **Vector Store**: Create persistent vector database with ChromaDB
4. ✅ **Basic RAG**: Enable question-answering with citation tracking
5. ✅ **Data Persistence**: Save all processed data for reproducibility

### Success Criteria
- ✅ Fetch 100+ ArXiv papers successfully
- ✅ Process documents with proper metadata preservation
- ✅ Generate embeddings for all document chunks
- ✅ Create persistent vector store
- ✅ All 55+ unit and integration tests passing
- ✅ Complete documentation and guides

---

## 🏗️ Architecture

### System Components

```
┌─────────────────┐
│  ArXiv Fetcher  │ → Fetches metadata and PDFs
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Document Loader │ → Combines metadata + PDF text
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Document Chunker│ → Splits into manageable chunks
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Embedder      │ → Generates vector embeddings
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Vector Store   │ → ChromaDB for similarity search
└─────────────────┘
```

### Data Flow

1. **Fetch**: `scripts/fetch_arxiv_data.py` → `data/raw/arxiv_metadata/`
2. **Load**: Metadata + PDFs → `Document` objects
3. **Chunk**: Documents → Smaller chunks with metadata
4. **Embed**: Chunks → Vector embeddings (384-dim)
5. **Store**: Embeddings → ChromaDB vector store

---

## 🚀 Getting Started

### Prerequisites

```bash
# Python 3.10+
python --version

# Virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 1: Fetch ArXiv Data

**⚠️ CRITICAL**: You must fetch data before running the embedding pipeline. No data = No chunks = No embeddings!

```bash
# Fetch 100 papers (metadata only)
python scripts/fetch_arxiv_data.py --max-results 100

# Fetch 500 papers with PDFs (for full-text processing)
python scripts/fetch_arxiv_data.py --max-results 500 --download-pdfs

# Fetch in batches (recommended for large datasets)
python scripts/fetch_arxiv_data.py --batches 5 --results-per-batch 100 --delay 5.0
```

**What this does**:
- Fetches metadata from ArXiv API
- Saves metadata to `data/raw/arxiv_metadata/<arxiv_id>.json`
- Optionally downloads PDFs to `data/raw/arxiv_pdfs/`
- Respects rate limits (5 second delay between batches)

**Expected Output**:
```
[INFO] [arxiv_fetcher] Starting batched ArXiv fetch
[INFO] [arxiv_fetcher] Fetched ArXiv batch
[INFO] [arxiv_fetcher] Saving ArXiv metadatas to disk
[INFO] [arxiv_fetcher] Saved ArXiv metadata
```

### Step 2: Run Embedding Pipeline

#### Option A: Using Jupyter Notebook (Recommended for Exploration)

```bash
# Start Jupyter
jupyter notebook notebooks/02_embedding_pipeline.ipynb
```

The notebook includes:
- ✅ Data fetching instructions
- ✅ Step-by-step processing
- ✅ Error handling and validation
- ✅ Results visualization

#### Option B: Using Python Script

```python
from src.embedding.document_processor import process_arxiv_abstracts

# Process abstracts only (faster, no PDFs needed)
documents = process_arxiv_abstracts(max_documents=100)

# Or process full papers (requires PDFs)
from src.embedding.document_processor import DocumentProcessor

processor = DocumentProcessor()
documents = processor.process_documents(
    include_full_text=True,
    max_documents=100,
    save_to_disk=True
)
```

### Step 3: Verify Results

```bash
# Check processed data
ls -lh data/processed/arxiv/

# Check vector store
ls -lh vector_db/

# Run tests
pytest tests/ -v
```

---

## 📁 Data Structure

### Raw Data (`data/raw/`)

```
data/raw/
├── arxiv_metadata/
│   ├── 1808.01591v1.json    # Individual metadata files
│   ├── 2001.00001v1.json
│   └── ...
└── arxiv_pdfs/              # Optional: PDF files
    ├── 1808.01591v1.pdf
    └── ...
```

**Metadata File Format**:
```json
{
  "id": "1808.01591v1",
  "title": "LISA: Explaining Recurrent Neural Network Judgments",
  "summary": "Recurrent neural networks...",
  "authors": ["Pankaj Gupta", "Hinrich Schütze"],
  "published": "2018-08-05T00:00:00Z",
  "updated": "2018-08-05T00:00:00Z",
  "links": ["https://arxiv.org/abs/1808.01591v1"],
  "categories": ["cs.CL", "cs.AI"],
  "pdf_url": "https://arxiv.org/pdf/1808.01591v1.pdf",
  "fetched_at": "2024-01-15T10:00:00Z",
  "source": "arxiv"
}
```

### Processed Data (`data/processed/`)

```
data/processed/
├── arxiv/
│   ├── documents/           # Individual loaded documents
│   │   ├── 1808.01591v1_2024-01-15_10-00-00_abstract_only.json
│   │   └── ...
│   └── chunks/              # Processed chunks with embeddings
│       ├── arxiv_chunks_2024-01-15_10-00-00_abstracts.json
│       └── arxiv_chunks_2024-01-15_10-00-00_full_text.json
```

**Chunk File Format**:
```json
{
  "source": "arxiv",
  "chunks": [
    {
      "chunk_id": "1808.01591v1_chunk_0",
      "chunk_index": 0,
      "page_content": "Title: LISA...",
      "metadata": {
        "arxiv_id": "1808.01591v1",
        "title": "LISA: Explaining...",
        "authors": ["Pankaj Gupta", "Hinrich Schütze"],
        "published": "2018-08-05T00:00:00Z",
        "pdf_url": "https://arxiv.org/pdf/1808.01591v1.pdf",
        "source": "arxiv",
        "embedding": [0.123, -0.456, ...],  # 384-dim vector
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
      }
    }
  ]
}
```

---

## 🔧 Configuration

### Environment Variables (`.env`)

```bash
# Optional: OpenAI API key (for future LLM integration)
OPENAI_API_KEY=sk-...

# Optional: Anthropic API key
ANTHROPIC_API_KEY=sk-ant-...
```

### Config File (`src/config.py`)

Key settings:
- `RAW_DATA_DIR`: `data/raw/`
- `PROCESSED_DATA_DIR`: `data/processed/`
- `VECTOR_DB_DIR`: `vector_db/`
- `CHUNK_SIZE`: 1000 (characters)
- `CHUNK_OVERLAP`: 200 (characters)

---

## 📊 Key Components

### 1. ArXiv Fetcher (`src/ingestion/arxiv_fetcher.py`)

**Functions**:
- `ArxivFetcher(max_results=100)`: Fetch single batch
- `batch_fetch_arxiv_metadata(...)`: Fetch multiple batches
- `save_metadatas(metadatas)`: Save to disk
- `download_pdfs_from_metadatas_file()`: Download PDFs

**Usage**:
```python
from src.ingestion.arxiv_fetcher import (
    batch_fetch_arxiv_metadata,
    save_metadatas
)

# Fetch metadata
metadata = batch_fetch_arxiv_metadata(
    num_batches=5,
    results_per_batch=100,
    delay_seconds=5.0
)

# Save to disk
save_metadatas(metadata)
```

### 2. Document Loader (`src/ingestion/document_loader.py`)

**Class**: `DocumentLoader`

**Methods**:
- `load_metadata(arxiv_id)`: Load single metadata file
- `load_all_documents(include_full_text=False, max_documents=None)`: Load all documents
- `_save_loaded_documents(documents, include_full_text)`: Save loaded documents

**Usage**:
```python
from src.ingestion.document_loader import DocumentLoader

loader = DocumentLoader()

# Load abstracts only (no PDFs needed)
documents = loader.load_all_documents(
    include_full_text=False,
    max_documents=100
)

# Load full papers (requires PDFs)
documents = loader.load_all_documents(
    include_full_text=True,
    max_documents=100,
    save_to_disk=True  # Save individual documents
)
```

### 3. Document Chunker (`src/embedding/chunking.py`)

**Class**: `DocumentChunker`

**Strategies**:
- `ChunkingStrategy.NONE`: No chunking (use entire document)
- `ChunkingStrategy.RECURSIVE`: Recursive character splitting
- `ChunkingStrategy.TOKEN`: Token-based splitting

**Usage**:
```python
from src.embedding.chunking import DocumentChunker, ChunkingStrategy

chunker = DocumentChunker(
    strategy=ChunkingStrategy.RECURSIVE,
    chunk_size=1000,
    chunk_overlap=200
)

chunks = chunker.chunk_documents(documents)
```

### 4. Document Embedder (`src/embedding/embedder.py`)

**Class**: `DocumentEmbedder`

**Models**:
- Default: `all-MiniLM-L6-v2` (384 dimensions)
- Fast, CPU-friendly embeddings

**Usage**:
```python
from src.embedding.embedder import DocumentEmbedder

embedder = DocumentEmbedder(model_name="all-MiniLM-L6-v2")
embedded_docs = embedder.embed_documents(documents, batch_size=32)
```

### 5. Document Processor (`src/embedding/document_processor.py`)

**Class**: `DocumentProcessor` - Orchestrates the full pipeline

**Convenience Function**: `process_arxiv_abstracts(max_documents=None)`

**Usage**:
```python
from src.embedding.document_processor import (
    DocumentProcessor,
    process_arxiv_abstracts
)

# Quick way: Process abstracts
docs = process_arxiv_abstracts(max_documents=100)

# Full control: Custom processing
processor = DocumentProcessor(
    embedding_model="all-MiniLM-L6-v2",
    chunk_strategy=ChunkingStrategy.NONE
)

docs = processor.process_documents(
    include_full_text=False,
    max_documents=100,
    batch_size=32,
    save_to_disk=True
)
```

---

## 🧪 Testing

### Run All Tests

```bash
# Activate virtual environment
source .venv/bin/activate

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Test Structure

```
tests/
├── conftest.py                    # Shared fixtures
├── test_arxiv_fetcher.py          # ArXiv fetching tests
├── test_document_loader.py        # Document loading tests
├── test_chunking.py               # Chunking tests
├── test_embedder.py               # Embedding tests
├── test_document_processor.py    # Pipeline tests
├── test_chunk_saver.py            # Data persistence tests
├── test_integration.py            # End-to-end tests
└── test_setup.py                  # Project setup tests
```

### Test Coverage

- ✅ **Unit Tests**: 45+ tests covering individual components
- ✅ **Integration Tests**: 10+ tests for end-to-end workflows
- ✅ **Coverage**: >70% code coverage

---

## 🐛 Troubleshooting

### Problem: "No documents were loaded"

**Cause**: No metadata files in `data/raw/arxiv_metadata/`

**Solution**:
```bash
# Fetch metadata first
python scripts/fetch_arxiv_data.py --max-results 100
```

### Problem: "Metadata not found for <arxiv_id>"

**Cause**: Metadata file doesn't exist or wrong path

**Solution**:
```bash
# Check if metadata exists
ls data/raw/arxiv_metadata/<arxiv_id>.json

# Re-fetch if missing
python scripts/fetch_arxiv_data.py --max-results 100
```

### Problem: "No PDFs found" (when using full_text=True)

**Cause**: PDFs not downloaded

**Solution**:
```bash
# Download PDFs
python scripts/fetch_arxiv_data.py --max-results 100 --download-pdfs
```

### Problem: Rate limiting errors

**Cause**: Too many API calls too quickly

**Solution**:
```bash
# Use batch fetching with delays
python scripts/fetch_arxiv_data.py --batches 5 --results-per-batch 100 --delay 5.0
```

### Problem: Embedding model download fails

**Cause**: Network issues or disk space

**Solution**:
```bash
# Check disk space
df -h

# Try again (model is cached after first download)
python -c "from src.embedding.embedder import DocumentEmbedder; DocumentEmbedder()"
```

---

## 📈 Performance Benchmarks

### Phase 1 Metrics

| Operation | Time | Notes |
|-----------|------|-------|
| Fetch 100 papers | ~30s | With 5s delay between batches |
| Load 100 abstracts | ~1s | No PDF processing |
| Load 100 full papers | ~5-10min | PDF extraction is slow |
| Chunk 100 documents | ~2s | Recursive strategy |
| Embed 100 chunks | ~10s | CPU, batch_size=32 |
| Full pipeline (abstracts) | ~15s | End-to-end |
| Full pipeline (full text) | ~6-12min | End-to-end with PDFs |

### Resource Usage

- **Memory**: ~500MB-1GB (depending on batch size)
- **Disk**: ~50MB per 100 papers (metadata only), ~500MB with PDFs
- **CPU**: Single-threaded (CPU-friendly embeddings)

---

## ✅ Phase 1 Completion Checklist

- [x] ArXiv fetcher with rate limiting
- [x] Document loader (abstracts + full text)
- [x] Document chunker (multiple strategies)
- [x] Document embedder (HuggingFace models)
- [x] Document processor (orchestration)
- [x] Data persistence (save/load chunks)
- [x] Vector store integration (ChromaDB)
- [x] Comprehensive test suite (55+ tests)
- [x] Documentation and guides
- [x] Error handling and logging
- [x] Constants and enums for maintainability

---

## 🔄 Next Steps: Phase 2

Phase 2 will add:
- Multi-source integration (Semantic Scholar, PubMed)
- Router agent for source selection
- Enhanced metadata handling
- Cross-source citation tracking

See `docs/phase_guides/phase2.md` (coming soon) for details.

---

## 📚 Additional Resources

- [AGENT.md](../AGENT.md) - Architecture rules and guidelines
- [README.md](../../README.md) - Project overview
- [API Reference](api_reference.md) - Detailed API documentation
- [Scripts README](../../scripts/README.md) - Script usage guide

---

## 🤝 Support

For issues or questions:
1. Check this guide first
2. Review error logs in `logs/`
3. Run tests to verify setup
4. Check GitHub issues

---

**Last Updated**: December 2024  
**Phase Status**: ✅ Complete

