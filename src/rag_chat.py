import os
import re
import requests
import chromadb
from chromadb.utils import embedding_functions
from pdf_processor import process_paper, normalize_pdf_url
import config

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(PROJECT_ROOT, "chroma_db")

# Initialize ChromaDB client
client = chromadb.PersistentClient(path=CHROMA_DIR)
embedding_func = embedding_functions.DefaultEmbeddingFunction()

SYSTEM_PROMPT = """You are a helpful research paper assistant.
Your task is to answer questions about the paper based ONLY on the provided context.
If the context doesn't contain enough information to answer the question, say so.
Be precise and cite specific details from the paper when possible.

Context from the paper:
{context}

Question: {question}

Answer:"""


def get_collection_name(paper_id: str) -> str:
    """Generate a valid, safe ChromaDB collection name."""
    clean_id = re.sub(r'[^a-zA-Z0-9_-]', '_', str(paper_id))
    # ChromaDB collection names must be 3-63 characters
    name = f"paper_{clean_id}"
    if len(name) > 63:
        name = name[:63]
    return name


def init_paper_context(paper_id: str, pdf_url: str) -> str:
    """
    Download, extract, and store paper chunks in ChromaDB.
    Safe against re-initialization and duplicate collection names.
    """
    if not paper_id:
        raise ValueError("paper_id is required")

    collection_name = get_collection_name(paper_id)

    # 1. Process paper to get chunks
    try:
        chunks = process_paper(pdf_url)
    except Exception as e:
        # Fallback to abstract if available in papers.csv
        try:
            import pandas as pd
            csv_path = os.path.join(PROJECT_ROOT, "data", "papers.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                match = df[df["paper_id"].astype(str) == str(paper_id)]
                if not match.empty:
                    title = str(match.iloc[0].get("title", ""))
                    abstract = str(match.iloc[0].get("abstract", ""))
                    fallback_text = f"Title: {title}\n\nAbstract: {abstract}"
                    from pdf_processor import chunk_text
                    chunks = chunk_text(fallback_text)
                else:
                    raise e
            else:
                raise e
        except Exception:
            raise ValueError(f"Failed to process paper PDF ({str(e)})")

    if not chunks:
        raise ValueError("No extractable text chunks could be generated for this paper.")

    # 2. Reset collection for this paper to ensure clean state
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        embedding_function=embedding_func
    )

    # 3. Store chunks using upsert in safe batches
    documents = [c["text"] for c in chunks]
    metadatas = [{"start": c.get("start", 0), "end": c.get("end", 0)} for c in chunks]
    ids = [f"{paper_id}_{i}" for i in range(len(chunks))]

    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        collection.upsert(
            documents=documents[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size],
            ids=ids[i:i + batch_size]
        )

    return f"Context initialized for {paper_id} with {len(chunks)} passages."


def call_llm(prompt: str) -> str:
    """
    Call an external LLM endpoint if configured (OpenAI-compatible format).
    Returns synthesized answer or helpful configuration message.
    """
    cfg = config.get_llm_config() if hasattr(config, "get_llm_config") else getattr(config, "LLM_CONFIG", {})
    base_url = (cfg.get("base_url") or "").strip().rstrip("/")
    api_key = (cfg.get("api_key") or "").strip()
    model = (cfg.get("model") or "gpt-4o-mini").strip()

    if not api_key:
        return (
            "⚠️ **No LLM API key configured for answer synthesis.**\n\n"
            "To have AI generate synthesized answers for this paper, please add your API key in the `.env` file in the project folder.\n\n"
            "Supported keys in `.env`:\n"
            "• `GROQ_API_KEY=gsk_...` (Free fast inference at console.groq.com)\n"
            "• `OPENAI_API_KEY=sk-...`\n"
            "• `GEMINI_API_KEY=AIza...`\n"
            "• `OPENROUTER_API_KEY=sk-or-...`\n\n"
            "*(See `.env.example` in the project root for guidance.)*"
        )

    # Construct completions endpoint
    endpoint = base_url
    if not endpoint.endswith("/chat/completions"):
        if endpoint.endswith("/v1"):
            endpoint = f"{endpoint}/chat/completions"
        else:
            endpoint = f"{endpoint}/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert AI research assistant. "
                    "Answer user questions accurately and concisely based strictly on the provided paper passages. "
                    "Cite relevant sections or details from the text where appropriate."
                )
            },
            {"role": "user", "content": prompt}
        ],
        "temperature": cfg.get("temperature", 0.7),
        "max_tokens": cfg.get("max_tokens", 1000)
    }

    try:
        resp = requests.post(endpoint, headers=headers, json=payload, timeout=25)
        if resp.status_code == 200:
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()

        # Parse API error details if available
        try:
            err_data = resp.json()
            err_msg = err_data.get("error", {}).get("message") or err_data.get("message") or str(err_data)
        except Exception:
            err_msg = resp.text[:200]

        return f"⚠️ **LLM API Error ({resp.status_code}):** {err_msg}\n\nPlease check your API key and model settings in `.env`."
    except requests.exceptions.Timeout:
        return f"⚠️ **LLM Request Timed Out:** The request to `{endpoint}` took longer than 25 seconds."
    except Exception as e:
        return f"⚠️ **Could not connect to LLM at `{endpoint}`:** {str(e)}"


def format_extractive_answer(query: str, chunks: list[str]) -> str:
    """Format relevant passages from the paper when LLM is unavailable."""
    sections = []
    for idx, chunk in enumerate(chunks, start=1):
        clean_chunk = chunk.strip().replace("\n", " ")
        if len(clean_chunk) > 400:
            clean_chunk = clean_chunk[:400] + "..."
        sections.append(f"Passage {idx}:\n\"{clean_chunk}\"")

    combined = "\n\n".join(sections)
    return (
        f"Relevant findings extracted from the paper:\n\n"
        f"{combined}\n\n"
        f"*(Direct passages retrieved from the indexed PDF.)*"
    )


def chat_with_paper(paper_id: str, query: str) -> str:
    """
    Retrieve relevant paper passages and generate an answer.
    """
    if not paper_id or not query:
        raise ValueError("paper_id and query are required")

    collection_name = get_collection_name(paper_id)
    try:
        collection = client.get_collection(
            name=collection_name,
            embedding_function=embedding_func
        )
    except Exception:
        raise ValueError("Paper session not found. Please click 'Chat with paper' to initialize the paper.")

    total_chunks = collection.count()
    if total_chunks == 0:
        return "This paper has no indexed passages to query."

    # Retrieve relevant chunks
    n_results = min(4, total_chunks)
    results = collection.query(query_texts=[query], n_results=n_results)

    retrieved_chunks = results.get("documents", [[]])[0]
    if not retrieved_chunks:
        return "I could not find any relevant information in this paper to answer your question."

    # Build prompt
    context_text = "\n\n---\n\n".join(retrieved_chunks)
    prompt = SYSTEM_PROMPT.format(context=context_text, question=query)

    # Attempt LLM call
    llm_response = call_llm(prompt)
    if llm_response:
        return llm_response

    # Fallback to clear extractive summary
    return format_extractive_answer(query, retrieved_chunks)


def cleanup_paper_context(paper_id: str) -> str:
    """
    Delete the ChromaDB collection for this paper.
    """
    collection_name = get_collection_name(paper_id)
    try:
        client.delete_collection(name=collection_name)
        return f"Collection {collection_name} deleted."
    except Exception:
        return "Collection not found or already deleted."
