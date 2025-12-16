# TODO - Multi-Source Research Intelligence Platform

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
