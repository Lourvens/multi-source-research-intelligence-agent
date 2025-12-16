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
