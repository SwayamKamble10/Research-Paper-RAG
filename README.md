# Academic Paper Search — Hybrid Retrieval System

A research paper search application combining BM25 lexical search with dense vector embeddings using Reciprocal Rank Fusion.

## Project Structure

```
Research Paper search/
├── app.py                    # Flask API server
├── requirements.txt          # Python dependencies
├── src/
│   ├── hybrid_search.py      # Main hybrid search module (BM25 + Dense RRF)
│   ├── bm25_search.py        # BM25 lexical retrieval
│   ├── dense_search.py       # Dense vector retrieval (FAISS + sentence-transformers)
│   ├── data/
│   │   └── papers.csv        # Paper metadata
│   ├── indexes/              # FAISS index files
│   └── chroma_db/            # ChromaDB persistent storage
├── templates/
│   └── index.html            # Frontend HTML template
└── static/
    ├── css/styles.css        # Frontend styles
    └── js/app.js             # Frontend JavaScript
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the Flask Server

```bash
cd src
python app.py
```
cdngs

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve the search UI |
| `/api/search` | POST | Search papers |
| `/api/health` | GET | Health check |

### Search API Request

```json
POST /api/search
{
  "query": "large language model reasoning",
  "top_k": 10
}
```

### Search API Response

```json
{
  "results": [
    {
      "rank": 1,
      "paper_id": "2608.24188v1",
      "title": "Paritok-4B: Intent-Conditioned Context Compression...",
      "abstract": "Coding agents re-send large file reads...",
      "authors": "Jiayu Shi; Luzhuo Chen",
      "published": "2026-08-25",
      "categories": "cs.AI; cs.CL; cs.LG; cs.SE",
      "url": "https://arxiv.org/abs/2608.24188v1",
      "rrf_score": 0.023456,
      "bm25_rank": 2,
      "dense_rank": 1
    }
  ],
  "total": 10
}
```

## Integration Steps

### Development Mode (Current)

1. **Run the Flask server**:
   ```bash
   python app.py
   ```

2. **Access the UI** at `http://localhost:5000`

### Production Deployment

#### Option A: Running Behind a Reverse Proxy (Nginx)

1. Update `app.py` to run on localhost only:
   ```python
   if __name__ == '__main__':
       app.run(debug=False, host='127.0.0.1', port=5000)
   ```

2. Configure Nginx to proxy requests:
   ```nginx
   location / {
       proxy_pass http://127.0.0.1:5000;
       proxy_set_header Host $host;
       proxy_set_header X-Real-IP $remote_addr;
   }
   ```

#### Option B: Container Deployment (Docker)

Create a `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000
CMD ["python", "app.py"]
```

Build and run:
```bash
docker build -t paper-search .
docker run -p 5000:5000 paper-search
```

#### Option C: Deploy to Cloud (Railway/Render/Heroku)

1. Push your code to GitHub
2. Connect the repository to your chosen platform
3. Set the build command: `pip install -r requirements.txt`
4. Set the start command: `python app.py`
5. The platform will provide a public URL

### Adding More Papers

To add new papers to the search index:

1. Add papers to `src/data/papers.csv` with columns:
   - `paper_id`, `title`, `abstract`, `authors`, `published`, `categories`, `url`

2. Rebuild the dense index (if needed):
   ```python
   from src.dense_search import build_index
   build_index()
   ```

## How Hybrid Search Works

1. **BM25 Search**: Traditional keyword-based retrieval using Okapi BM25
2. **Dense Search**: Semantic similarity using sentence-transformers embeddings
3. **Reciprocal Rank Fusion**: Combines both rankings using the formula:
   ```
   RRF_score(d) = Σ 1 / (k + rank(d))
   ```
   where k=60 (constant) and rank(d) is the position in each ranking list

## Dependencies

- **flask** + **flask-cors**: Web server
- **rank-bm25**: BM25 implementation
- **sentence-transformers**: Dense embeddings
- **faiss-cpu**: Vector similarity search
- **chromadb**: Persistent vector storage (for dense search)
- **pandas**, **numpy**: Data handling"# Research-Paper-RAG" 
