# AGENT.md - Multi-Source Research Intelligence Platform

## 📋 Project Overview

**Project Name**: Multi-Source Research Intelligence Platform (MSRIP)

**Version**: 1.0.0 - Phase 1 (Foundation)

**Purpose**: Build an agentic RAG system that autonomously searches academic papers, patents, news, and datasets to generate comprehensive research reports with proper citations and contradiction analysis.

**Current Phase**: Phase 1 - Foundation & Basic RAG Setup

---

## 🎯 Project Vision & Goals

### Primary Objectives
1. **Autonomous Research**: Enable researchers to get comprehensive answers by querying multiple authoritative sources simultaneously
2. **Citation Integrity**: Every claim must be traceable to its source with proper academic citation
3. **Contradiction Detection**: Identify conflicting information across sources and present balanced analysis
4. **Scalability**: Design for future expansion from 1 source (Phase 1) to 10+ sources (Phase 10)

### Success Metrics
- **Retrieval Relevance**: >80% of retrieved documents rated as relevant
- **Answer Accuracy**: >85% factually correct (human evaluation)
- **Citation Coverage**: 100% of claims cited to source documents
- **Response Time**: <60 seconds for complex multi-source queries (Phase 8+)
- **User Satisfaction**: >4.0/5.0 rating from test users

---

## 🏗️ System Architecture Principles

### 1. **Modularity First**
- Each component (fetcher, processor, vector store, agent) is independently testable
- Interfaces are clearly defined between components
- New data sources can be added without modifying existing code

### 2. **Progressive Complexity**
- Phase 1: Single source (ArXiv), basic RAG
- Phase 2-3: Multi-source, routing logic
- Phase 4-6: Agents, self-reflection, query rewriting
- Phase 7-10: Orchestration, synthesis, production deployment

### 3. **Data-Centric Design**
- Rich metadata preservation at every stage
- Traceability from answer → chunk → source document
- Versioning of processed data for reproducibility

### 4. **Agent Autonomy**
- Agents make decisions based on query analysis
- Self-evaluation and iterative improvement
- Fallback mechanisms for failure scenarios

---

## 📐 Architectural Rules & Guidelines

### Rule 1: **Single Source of Truth**
- Raw data from APIs is saved immediately (JSON format)
- All processing derives from these raw files
- Never modify raw data; create processed versions

**Example**:
```
data/
├── raw/
│   └── arxiv_papers_2024-01-15.json  # Never modified
└── processed/
    └── arxiv_chunks_2024-01-15.json   # Derived from raw
```

### Rule 2: **Metadata is Sacred**
Every document chunk MUST contain:
- `source`: Origin (arxiv, pubmed, news, etc.)
- `paper_id`: Unique identifier
- `title`: Document title
- `authors`: Creator attribution
- `published`: Publication date (ISO 8601 format)
- `pdf_url` or `url`: Link to original source
- `chunk_id`: Unique chunk identifier
- `chunk_index`: Position in original document

**Why**: Citations and source verification depend on complete metadata

### Rule 3: **Idempotency & Reproducibility**
- Running the same pipeline twice with same input produces identical output
- All random seeds are fixed
- API responses are cached when possible
- Version control includes data provenance

### Rule 4: **Rate Limiting & API Respect**
- Implement delays between API calls (3-5 seconds)
- Respect API rate limits explicitly
- Cache API responses to avoid redundant calls
- Use exponential backoff for retries

**Implementation**:
```python
# Good
time.sleep(3)  # Be nice to API servers
result = fetch_with_retry(url, max_retries=3, backoff=2)

# Bad
for i in range(1000):
    fetch(url)  # Will get rate limited or banned
```

### Rule 5: **Error Handling is Non-Negotiable**
Every external call MUST have:
- Try-except blocks with specific exception types
- Logging of errors with context
- Graceful degradation when possible
- User-friendly error messages

**Template**:
```python
try:
    result = api_call()
except RateLimitError as e:
    logger.warning(f"Rate limited: {e}. Waiting 60s...")
    time.sleep(60)
    result = api_call()
except APIError as e:
    logger.error(f"API error: {e}")
    return fallback_result()
except Exception as e:
    logger.critical(f"Unexpected error: {e}")
    raise
```

### Rule 6: **Logging Standards**
All components must log:
- INFO: Normal operations (papers fetched, chunks created)
- WARNING: Degraded performance (rate limits, retries)
- ERROR: Failures with recovery attempts
- CRITICAL: Unrecoverable failures

**Format**: `[TIMESTAMP] [LEVEL] [COMPONENT] Message`

### Rule 7: **Configuration Over Hardcoding**
All configurable values go in:
- `.env` for secrets (API keys)
- `config.yaml` for parameters (chunk size, model names)
- Never hardcode API keys, file paths, or magic numbers

### Rule 8: **Testing at Every Phase**
Before moving to next phase:
- ✅ Unit tests for each function
- ✅ Integration tests for pipeline
- ✅ Manual evaluation of output quality
- ✅ Performance benchmarks logged

### Rule 9: **Documentation is Development**
- Every function has docstring with Args, Returns, Raises
- README updated after each phase
- Architecture diagrams for complex workflows
- CHANGELOG.md tracks all changes

### Rule 10: **Privacy & Ethics**
- No storage of API keys in code or git
- User queries are not logged permanently (GDPR compliance)
- Fair use of academic papers (no full-text storage without permission)
- Attribution to all sources

---

## 🗂️ Project Structure Standards

```
research-rag/
├── .env                          # API keys (NEVER commit)
├── .env.example                  # Template for .env
├── .gitignore                    # Ignore secrets, data, vector_db
├── requirements.txt              # Python dependencies
├── config.yaml                   # Configuration parameters
├── README.md                     # User-facing documentation
├── AGENT.md                      # This file - rules and architecture
├── CHANGELOG.md                  # Version history
├── notebooks/                    # Jupyter notebooks for exploration
│   ├── phase1_basic_rag.ipynb
│   ├── phase2_multi_source.ipynb
│   └── experiments/              # Ad-hoc experiments
├── src/                          # Source code
│   ├── __init__.py
│   ├── config.py                 # Config loading utilities
│   ├── ingestion/                # Data fetching & processing
│   │   ├── __init__.py
│   │   ├── arxiv_fetcher.py
│   │   ├── pubmed_fetcher.py     # Phase 2+
│   │   ├── document_processor.py
│   │   └── base_fetcher.py       # Abstract base class
│   ├── vector_store/             # Vector database management
│   │   ├── __init__.py
│   │   ├── chroma_store.py
│   │   └── store_interface.py    # Abstract interface
│   ├── agents/                   # Agent implementations (Phase 3+)
│   │   ├── __init__.py
│   │   ├── router_agent.py
│   │   ├── grader_agent.py
│   │   ├── rewriter_agent.py
│   │   └── orchestrator.py       # Phase 8+
│   ├── rag/                      # RAG chains
│   │   ├── __init__.py
│   │   ├── basic_chain.py
│   │   └── agentic_chain.py      # Phase 4+
│   └── utils/                    # Utilities
│       ├── __init__.py
│       ├── logging_config.py
│       ├── metrics.py
│       └── citation_formatter.py
├── tests/                        # Test suite
│   ├── __init__.py
│   ├── test_arxiv_fetcher.py
│   ├── test_document_processor.py
│   ├── test_rag_chain.py
│   └── test_integration.py
├── data/                         # Data directory (not in git)
│   ├── raw/                      # Raw API responses
│   ├── processed/                # Processed documents
│   └── cache/                    # API response cache
├── vector_db/                    # Vector store (not in git)
├── logs/                         # Application logs (not in git)
├── scripts/                      # Utility scripts
│   ├── fetch_papers.py           # CLI for paper fetching
│   ├── create_vector_db.py       # CLI for vector DB creation
│   └── evaluate_retrieval.py    # Evaluation script
└── docs/                         # Extended documentation
    ├── phase_guides/
    │   ├── phase1.md
    │   └── phase2.md
    ├── api_reference.md
    └── architecture_diagrams/
```

---

## 🔒 Security & Secrets Management

### API Key Storage
**NEVER** commit these to git:
- OpenAI API keys
- Anthropic API keys
- Any authentication tokens
- Database credentials

**Correct approach**:
```bash
# .env file (in .gitignore)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

```python
# config.py
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment")
```

### .gitignore Requirements
```gitignore
# Secrets
.env
*.key
secrets/

# Data
data/
vector_db/
*.db

# Logs
logs/
*.log

# Python
__pycache__/
*.pyc
.pytest_cache/

# Jupyter
.ipynb_checkpoints/

# IDE
.vscode/
.idea/
```

---

## 📊 Data Management Rules

### Data Versioning
Raw data files include timestamps:
```
arxiv_papers_2024-01-15T14-30-00.json
pubmed_papers_2024-01-15T14-35-00.json
```

### Data Retention
- **Raw data**: Keep for 90 days minimum
- **Processed data**: Can be regenerated, keep for 30 days
- **Vector stores**: Persist indefinitely, version by date
- **Logs**: Keep for 30 days

### Data Size Limits
- Single file: <500MB
- Total raw data: <5GB per source
- Vector DB: <10GB for Phase 1-3
- If exceeded: Implement archival strategy

---

## 🧪 Quality Assurance Standards

### Code Quality
- **Line length**: Max 100 characters
- **Function length**: Max 50 lines (prefer smaller)
- **Cyclomatic complexity**: Max 10
- **Type hints**: Required for all function signatures
- **Docstrings**: Required for all public functions

### Testing Requirements
**Phase 1**:
- Unit test coverage: >70%
- Integration tests: 3+ end-to-end scenarios

**Phase 4+**:
- Unit test coverage: >80%
- Agent behavior tests
- Regression tests for known edge cases

### Performance Benchmarks
Track these metrics:
- Paper fetch time: <2 seconds per paper
- Embedding time: <0.1 seconds per chunk
- Query time: <3 seconds (Phase 1)
- Memory usage: <2GB (Phase 1)

---

## 🔄 Development Workflow

### Phase Completion Checklist
Before marking a phase complete:
- [ ] All features implemented and tested
- [ ] Documentation updated (README, AGENT.md, docstrings)
- [ ] Manual testing completed
- [ ] Performance benchmarks recorded
- [ ] Code reviewed (self-review minimum)
- [ ] Git commit with descriptive message
- [ ] Tag release (e.g., `v1.0-phase1`)

### Git Commit Standards
Format: `[PHASE] [TYPE]: Description`

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `test`: Tests
- `refactor`: Code refactoring
- `perf`: Performance improvement

Examples:
```
[PHASE1] feat: Add ArXiv fetcher with rate limiting
[PHASE1] docs: Update AGENT.md with data rules
[PHASE2] feat: Integrate Semantic Scholar API
```

### Branch Strategy (Optional for Solo Project)
- `main`: Production-ready code
- `develop`: Integration branch
- `feature/phase-X-description`: Feature branches

---

## 🎓 Learning & Iteration Rules

### Rule of Progressive Refinement
1. **Make it work** (Phase 1-2)
2. **Make it right** (Phase 3-5)
3. **Make it fast** (Phase 6-8)
4. **Make it production-ready** (Phase 9-10)

### When to Refactor
Refactor when you notice:
- Code duplication (>3 times)
- Functions >50 lines
- Complex nested conditionals (>3 levels)
- Hard-to-test code
- Performance bottlenecks

### Technical Debt Tracking
Maintain `TODO.md` with:
- Known issues
- Optimization opportunities  
- Future enhancements
- Technical debt items

Format:
```markdown
## Phase 1 TODOs
- [ ] Add batch processing for large paper sets
- [ ] Implement async API calls
- [x] Add retry logic with exponential backoff

## Technical Debt
- Document processor could use caching (not critical for Phase 1)
- Consider switching to async/await in Phase 3
```

---

## 🚨 Common Pitfalls to Avoid

### 1. **Over-Engineering Early**
❌ Don't build abstract base classes in Phase 1
✅ Start simple, refactor when you add 2nd implementation

### 2. **Ignoring Rate Limits**
❌ Hammering APIs without delays
✅ Respect rate limits, implement exponential backoff

### 3. **Poor Chunk Boundaries**
❌ Splitting mid-sentence
✅ Use RecursiveCharacterTextSplitter with semantic separators

### 4. **Metadata Loss**
❌ Losing source information during processing
✅ Propagate metadata through entire pipeline

### 5. **Premature Optimization**
❌ Optimizing before profiling
✅ Measure first, optimize bottlenecks only

### 6. **Insufficient Logging**
❌ Silent failures
✅ Log at appropriate levels, make debugging easy

### 7. **Hardcoded Paths**
❌ `open('/Users/me/data/papers.json')`
✅ Use `pathlib.Path` and config files

### 8. **No Error Recovery**
❌ Crash on first API failure
✅ Retry with backoff, graceful degradation

---

## 📈 Monitoring & Observability

### Metrics to Track (Per Phase)

**Phase 1**:
- Papers fetched (count, source)
- Chunks created (count, avg size)
- Embedding time
- Query latency
- Retrieval precision (manual eval)

**Phase 4+**:
- Agent decision distribution
- Query rewrite frequency
- Document grading scores
- Contradiction detection rate

**Phase 8+**:
- End-to-end latency
- API costs per query
- User satisfaction scores
- Error rates by component

### Logging Implementation
```python
import logging
from datetime import datetime

def setup_logging(component_name: str):
    """Standard logging configuration"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{component_name}_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(component_name)
```

---

## 🤝 Collaboration Guidelines (Future)

### Code Review Standards
- Check for rule compliance (this document)
- Verify tests pass
- Validate documentation
- Test manually if UI changes

### Communication
- Document design decisions in code comments
- Update AGENT.md when changing architecture
- Create issues for bugs/features
- Use clear commit messages

---

## 📚 References & Resources

### Key Documentation
- [LangChain Docs](https://python.langchain.com/)
- [ArXiv API Guide](https://info.arxiv.org/help/api/index.html)
- [ChromaDB Docs](https://docs.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)

### Research Papers
- "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (RAG paper)
- "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection"
- "Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG"

### Best Practices
- [OpenAI RAG Best Practices](https://platform.openai.com/docs/guides/prompt-engineering)
- [LangChain RAG Patterns](https://python.langchain.com/docs/use_cases/question_answering/)

---

## 📝 Version History

### v1.0.0-phase1 (Current)
- Initial project setup
- ArXiv integration
- Basic RAG pipeline
- Vector store with ChromaDB
- Simple Q&A interface

### Planned Versions
- v1.1.0-phase2: Multi-source integration
- v1.2.0-phase3: Router agent
- v1.3.0-phase4: Document grading
- v2.0.0-phase8: Full orchestration

---

## 🎯 Success Definition

**Phase 1 is complete when**:
- ✅ 100+ ArXiv papers ingested
- ✅ Vector store created and persistent
- ✅ Query returns relevant results >70% of time
- ✅ Citations traceable to source
- ✅ All tests passing
- ✅ Documentation complete

**Project is complete when (Phase 10)**:
- 🎯 Handles 10+ data sources
- 🎯 Detects contradictions automatically
- 🎯 Generates publication-ready reports
- 🎯 Production deployed with auth
- 🎯 <60s response time
- 🎯 >85% user satisfaction

---

## 🔄 Continuous Improvement

This document evolves with the project. Update it when:
- Architecture changes significantly
- New patterns emerge
- Rules need clarification
- Best practices discovered

**Last Updated**: 2024-01-15
**Next Review**: After Phase 2 completion

---

## 📞 Support & Questions

For questions about this document or project architecture:
1. Review relevant phase guide in `docs/phase_guides/`
2. Check TODO.md for known issues
3. Consult referenced documentation
4. Document new patterns discovered

---

**Remember**: These rules exist to make development faster and more reliable in the long run. When in doubt, refer back to this document.

**Happy Building! 🚀**