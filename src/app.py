"""
Flask API for Academic Paper Hybrid Search
Exposes the hybrid search (BM25 + Dense) via REST endpoints
"""

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import sys
import os
import math

# Get the project root (parent of src/)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(PROJECT_ROOT)

# Add current directory (src) to path for imports
sys.path.insert(0, PROJECT_ROOT)

# Import modules directly from src
from hybrid_search import hybrid_search
import rag_chat

app = Flask(__name__,
    static_folder=os.path.join(BASE_DIR, 'static'),
    template_folder=os.path.join(BASE_DIR, 'templates'))
CORS(app)


@app.route('/')
def index():
    """Serve the main search page"""
    return render_template('index.html')


@app.route('/api/search', methods=['POST'])
def search():
    """
    Search for papers using hybrid search (BM25 + Dense)

    Request body:
        {
            "query": "search query string",
            "top_k": 10 (optional, default 10)
        }

    Response:
        {
            "results": [
                {
                    "rank": 1,
                    "paper_id": "...",
                    "title": "...",
                    "abstract": "...",
                    "authors": "...",
                    "url": "...",
                    "rrf_score": 0.123,
                    "bm25_rank": 2,
                    "dense_rank": 5
                },
                ...
            ],
            "latency_ms": 123.45,
            "total": 10
        }
    """
    data = request.get_json()

    if not data or 'query' not in data:
        return jsonify({'error': 'Query parameter is required'}), 400

    query = data['query'].strip()

    if not query:
        return jsonify({'error': 'Query cannot be empty'}), 400

    top_k = data.get('top_k', 10)

    try:
        results = hybrid_search(
            query=query,
            top_k=top_k,
            candidate_k=100,
            rank_constant=60
        )

        # Convert DataFrame to list of dicts for JSON response
        # Helper to handle NaN values (not valid in JSON)
        def clean_value(val):
            if isinstance(val, float) and math.isnan(val):
                return None
            return val

        papers = []
        for rank, (_, row) in enumerate(results.iterrows(), start=1):
            paper = {
                'rank': rank,
                'paper_id': clean_value(row.get('paper_id', '')),
                'title': clean_value(row.get('title', 'Unknown')),
                'abstract': clean_value(row.get('abstract', '')),
                'authors': clean_value(row.get('authors', '')),
                'published': clean_value(row.get('published', '')),
                'categories': clean_value(row.get('categories', '')),
                'url': clean_value(row.get('url', '')),
                'rrf_score': round(float(row.get('rrf_score', 0)), 6),
                'bm25_rank': clean_value(row.get('bm25_rank')),
                'dense_rank': clean_value(row.get('dense_rank'))
            }
            papers.append(paper)

        return jsonify({
            'results': papers,
            'total': len(papers)
        })

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Search failed: {str(e)}'}), 500


@app.route('/api/chat/open', methods=['POST'])
def chat_open():
    """Initialize RAG context for a paper."""
    data = request.get_json() or {}
    paper_id = data.get('paper_id')
    pdf_url = data.get('pdf_url')

    if not paper_id:
        return jsonify({'error': 'Missing paper_id'}), 400
    if not pdf_url:
        pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"

    try:
        message = rag_chat.init_paper_context(paper_id, pdf_url)
        return jsonify({'success': True, 'message': message})
    except Exception as e:
        return jsonify({'error': f'Failed to initialize chat: {str(e)}'}), 500


@app.route('/api/chat/message', methods=['POST'])
def chat_message():
    """Handle chat queries."""
    data = request.get_json() or {}
    paper_id = data.get('paper_id')
    query = data.get('message')
    if not paper_id or not query:
        return jsonify({'error': 'Missing paper_id or message'}), 400
    try:
        response = rag_chat.chat_with_paper(paper_id, query)
        return jsonify({'success': True, 'response': response})
    except Exception as e:
        return jsonify({'error': f'Chat failed: {str(e)}'}), 500


@app.route('/api/chat/close', methods=['POST'])
def chat_close():
    """Cleanup chat context."""
    data = request.get_json()
    paper_id = data.get('paper_id')
    if not paper_id:
        return jsonify({'error': 'Missing paper_id'}), 400
    try:
        message = rag_chat.cleanup_paper_context(paper_id)
        return jsonify({'success': True, 'message': message})
    except Exception as e:
        return jsonify({'error': f'Cleanup failed: {str(e)}'}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'service': 'Paper Search API'})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
