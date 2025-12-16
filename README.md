# Multi-Source Research Intelligence Platform (MSRIP)

## 📚 Overview

An agentic RAG system that autonomously searches academic papers, patents, news, and datasets to generate comprehensive research reports with proper citations and contradiction analysis.

**Current Phase**: Phase 1 - Foundation & Basic RAG Setup

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

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

### 3. Fetch ArXiv Data

**⚠️ IMPORTANT**: You must fetch data before running the pipeline!

```bash
# Fetch 100 papers (metadata only - for abstracts)
python scripts/fetch_arxiv_data.py --max-results 100

# OR fetch with PDFs (for full-text processing)
python scripts/fetch_arxiv_data.py --max-results 100 --download-pdfs
```

### 4. Run Phase 1 Pipeline

```bash
# Option A: Using Jupyter notebook (recommended)
jupyter notebook notebooks/02_embedding_pipeline.ipynb

# Option B: Using Python script
python -c "from src.embedding.document_processor import process_arxiv_abstracts; process_arxiv_abstracts(max_documents=10)"
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

## 🎯 Phase 1 Features (✅ Complete)

- ✅ ArXiv paper fetching with rate limiting
- ✅ Document processing and chunking (abstracts + full text)
- ✅ Document embedding with HuggingFace models
- ✅ Data persistence (save/load processed chunks)
- ✅ Comprehensive test suite (55+ tests)
- ✅ Complete documentation and guides
- 🚧 Vector store with ChromaDB (in progress)
- 🚧 Basic RAG question-answering (in progress)

## 📖 Documentation

- **[Phase 1 Guide](docs/phase_guides/phase1.md)** - Complete Phase 1 implementation guide
- [AGENT.md](AGENT.md) - Architecture rules and guidelines (if exists)
- [Scripts README](scripts/README.md) - Utility script documentation

## 🧪 Testing

```bash
# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 📊 Current Capabilities (Phase 1)

**Data Sources**: ArXiv (academic papers)  
**Processing**: Document loading, chunking, and embedding  
**Embeddings**: 384-dimensional vectors (all-MiniLM-L6-v2)  
**Data Persistence**: JSON-based chunk storage  
**Test Coverage**: 55+ tests, >70% code coverage  
**Performance**: ~15s for 100 abstracts, ~5-10min for 100 full papers

## 🗺️ Roadmap

- [x] **Phase 1**: Foundation & Basic RAG Setup ✅
  - [x] ArXiv integration
  - [x] Document processing pipeline
  - [x] Embedding generation
  - [x] Data persistence
  - [x] Test suite
  - [ ] Vector store integration (in progress)
  - [ ] Basic RAG chain (in progress)
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

## 🙏 Acknowledgments

Built with:
- [LangChain](https://python.langchain.com/)
- [ChromaDB](https://www.trychroma.com/)
- [ArXiv API](https://info.arxiv.org/help/api/index.html)
- [Sentence Transformers](https://www.sbert.net/)

---

## 🆘 Troubleshooting

### "No documents were loaded"
**Solution**: Fetch ArXiv data first:
```bash
python scripts/fetch_arxiv_data.py --max-results 100
```

### "No metadata files found"
**Solution**: Check `data/raw/arxiv_metadata/` directory exists and contains JSON files.

### See [Phase 1 Guide](docs/phase_guides/phase1.md) for detailed troubleshooting.

---

**Last Updated**: December 2024 | **Version**: 1.0.0-phase1 | **Status**: ✅ Phase 1 Complete
