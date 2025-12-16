# Setup Instructions

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
venv\Scripts\activate
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
source venv/bin/activate  # or venv\Scripts\activate on Windows
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
