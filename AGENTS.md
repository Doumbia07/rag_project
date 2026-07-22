# AI Agent Guidelines for RAG Project

**Project**: Multi-Domain RAG Search Engine  
**Purpose**: Compare retrieval and ranking approaches (BM25, FAISS, Hybrid, Reranking, RAG) on BEIR datasets

## 🏗️ Architecture Overview

### Backend Structure (`backend/`)
- **app.py**: Flask API server with 5 search endpoints and lazy engine loading
- **Search Engines** (one class per approach):
  - `bm25.py`: Keyword-based search using BM25 algorithm
  - `faiss_search.py`: Semantic search using embeddings + FAISS index
  - `hybrid.py`: Weighted fusion of BM25 and FAISS scores
  - `rerank.py`: Cross-Encoder reranking (post-processing)
  - `rag.py`: Generative pipeline (retrieve + generate)
- **data_loader.py**: BEIR dataset handling (SciFact, NFCorpus)
- **evaluator.py**: Metrics computation (nDCG, Recall@k, latency)

### Frontend (`frontend/streamlit_app.py`)
- Streamlit UI for user queries, engine selection, result visualization

### Data Flow
```
User Query (French) → Streamlit UI → Flask API (/search_*)
→ Search Engine (BM25/FAISS/Hybrid/Rerank/RAG)
→ Corpus Search → Results (title, text, score) → JSON Response
```

## 🎯 Key Development Patterns

### Engine Implementation Pattern
Each search engine inherits from a common interface (or follows the same signature):
- `__init__(dataset_name, sample_size=None)`: Load corpus and initialize
- `search(query, top_k=10)`: Return list of dicts with keys: `doc_id`, `title`, `text`, `score`

Example:
```python
class NewSearch:
    def __init__(self, dataset_name="nfcorpus", sample_size=None):
        self.corpus, self.queries, self.qrels = load_dataset(dataset_name, sample_size)
    
    def search(self, query, top_k=10):
        # Return [{"doc_id": "...", "title": "...", "text": "...", "score": 0.95}, ...]
        pass
```

### API Endpoint Pattern (`app.py`)
- **Lazy loading**: Engines are initialized on first request (not at startup)
- **Common parameters**: `query`, `top_k` (default 10)
- **Response format**: JSON with `results` array and metadata (`latency_ms`, `engine`)
- **Language handling**: Translates French queries to English before search, with fallback dictionary

Example endpoints:
```
POST /search_bm25      → {"query": "...", "top_k": 10}
POST /search_faiss     → {"query": "...", "top_k": 10}
POST /search_hybrid    → {"query": "...", "top_k": 10}
POST /search_rerank    → {"query": "...", "top_k": 10}
POST /search_rag       → {"query": "...", "top_k": 10}
```

### Configuration Constants (`app.py`)
Located at top of file, adjusted per experiment:
- `DATASET`: "nfcorpus" or "scifact"
- `SAMPLE_SIZE`: Limit corpus size (None = full corpus)
- `ALPHA`: BM25/FAISS weight for hybrid (0.5-0.7 typical)
- `TOP_K_DEFAULT`: Results per query

### Tokenization
- Uses regex-based tokenization: `re.findall(r'\w+', text.lower())`
- Applied consistently across BM25 and hybrid approaches
- Handle empty/non-string inputs gracefully

### Data Loading
- **load_dataset(dataset_name, sample_size)** returns: `(corpus_dict, queries_dict, qrels_dict)`
- Corpus format: `{doc_id: {"text": "...", "title": "..."}}`
- Queries format: `{query_id: "query text"}`
- Qrels format: `{query_id: {doc_id: relevance_score}}`

## 🔑 Important Conventions & Pitfalls

| Convention | Importance | Notes |
|-----------|-----------|-------|
| **Lazy Engine Loading** | High | Call `get_engine(name)` in routes; don't init all at startup |
| **Search Result Structure** | High | Must return list of dicts with `doc_id`, `title`, `text`, `score` |
| **Tokenization Consistency** | Medium | Use same regex tokenization across engines for fair comparison |
| **Error Handling in Translation** | Medium | Use fallback dict if Google Translate fails (deep_translator) |
| **Score Normalization** | Medium | Hybrid engine normalizes scores to [0, 1] before weighting |
| **CORS Enabled** | Low | CORS headers already set (`Flask-CORS`); required for Streamlit frontend |

### Performance Considerations
1. **FAISS index size**: ~80 MB for SciFact (5,183 docs); loaded into memory
2. **Embedding generation**: Expensive; cached via FAISS or sentence-transformers
3. **Sample size**: Reduce `SAMPLE_SIZE` for faster iteration during development
4. **Cold start**: First engine request triggers lazy loading (~2-5 sec)

### Multilingual Handling
- Queries may come in French; translation to English is automatic
- Result text remains in original corpus language
- If translation fails: use hardcoded French→English dictionary fallback

## 🧪 Testing

### Test Structure (`tests/`)
- **test_api.py**: API endpoint tests (currently minimal)
- **test_bm25.py**: BM25 search engine unit tests

### Running Tests
```bash
pytest tests/             # All tests
pytest tests/test_bm25.py # Specific test file
pytest -v                 # Verbose output
```

### Test Patterns
- Use `load_dataset()` to get test corpus
- Verify `search()` returns correct structure
- Check relevance scores are in expected range

## 📊 Evaluation

- **evaluator.py**: Compute metrics (nDCG@k, Recall@k, MRR, latency)
- Qrels are loaded from `/data/{dataset}/qrels/` (train/dev/test splits)
- Compare engines using same qrels for fairness

## 🚀 Running the Project

### Backend (Flask)
```bash
python -m backend.app      # Or: FLASK_APP=backend.app flask run
```

### Frontend (Streamlit)
```bash
streamlit run frontend/streamlit_app.py
```

### Both (Local Development)
- Run Flask in one terminal: `python -m backend.app`
- Run Streamlit in another: `streamlit run frontend/streamlit_app.py`
- Streamlit connects to Flask API at `http://localhost:5000`

## 📝 Common Tasks & Patterns

### Adding a New Search Engine
1. Create `backend/new_engine.py` with `NewEngine(dataset_name, sample_size=None)`
2. Implement `search(query, top_k=10)` → returns list of result dicts
3. Add to `app.py`: import + add case in `get_engine()`
4. Add endpoint: `@app.route("/search_newengine", methods=["POST"])`
5. Test with: `curl -X POST http://localhost:5000/search_newengine -d '{"query":"...", "top_k":10}'`

### Debugging Engine Issues
- Check print statements in `__init__` (lazy loading confirmation)
- Verify corpus loaded: `print(len(self.corpus))`
- Test tokenization: `print(self._tokenize("test query"))`
- Monitor latency: Flask response includes `latency_ms`

### Changing Datasets
Edit `app.py` line: `DATASET = "nfcorpus"` → `"scifact"`  
Restart Flask. Corpus will auto-load on first request.

### Adjusting Hybrid Weight (Alpha)
Edit `app.py` line: `ALPHA = 0.6`  
- `0.5`: Equal BM25 + FAISS
- `0.7`: Favor BM25
- `0.3`: Favor FAISS

## 📚 Key Files to Know
- [README.md](README.md) — Project overview, installation, API docs
- [backend/app.py](backend/app.py) — Main Flask server + configuration
- [backend/data_loader.py](backend/data_loader.py) — BEIR dataset handling
- [backend/evaluator.py](backend/evaluator.py) — Metrics computation
- [requirements.txt](requirements.txt) — Python dependencies
- [Dockerfile](Dockerfile) — Docker deployment

## 🔗 Resources
- BEIR Benchmark: https://github.com/beir-cellar/beir
- FAISS Docs: https://github.com/facebookresearch/faiss
- Sentence-Transformers: https://www.sbert.net/
- BM25: https://en.wikipedia.org/wiki/Okapi_BM25
