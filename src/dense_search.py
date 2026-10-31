from pathlib import Path
import time

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


DATA_PATH = Path("data/papers.csv")
INDEX_DIR = Path("indexes")
INDEX_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDINGS_PATH = INDEX_DIR / "paper_embeddings.npy"
FAISS_PATH = INDEX_DIR / "papers.faiss"

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

def load_papers():
    papers=pd.read_csv(DATA_PATH).fillna("")
    papers=papers.drop_duplicates('paper_id')
    papers=papers.reset_index(drop=True)
    papers["search_text"] = (
            papers["title"].astype(str)
            + ". "
            + papers["abstract"].astype(str)
        )
    
    return papers


papers=load_papers()
model=SentenceTransformer(MODEL_NAME,device="cuda")

def build_index():
    documents = papers["search_text"].tolist()

    embeddings = model.encode(
        documents,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    # Inner product on normalized vectors = cosine similarity.
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    # res = faiss.StandardGpuResources()
    # index = faiss.index_cpu_to_gpu(res, 0, index)
    np.save(EMBEDDINGS_PATH, embeddings)
    faiss.write_index(index, str(FAISS_PATH))

    print(f"Indexed {index.ntotal} papers.")
    print(f"Embedding shape: {embeddings.shape}")


def load_index():
    embeddings = np.load(EMBEDDINGS_PATH)
    index = faiss.read_index(str(FAISS_PATH))

    if index.ntotal != len(papers):
        raise ValueError(
            "Index and paper metadata do not match. "
            "Rebuild the index using the current dataset."
        )

    if embeddings.shape[0] != len(papers):
        raise ValueError("Embedding count does not match papers.")

    return index


def dense_search(query, k=10):
    if not query.strip():
        return papers.iloc[0:0].copy()

    index = load_index()

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    start = time.perf_counter()
    scores, ids = index.search(query_embedding, k)
    elapsed_ms = (time.perf_counter() - start) * 1000

    valid = ids[0] >= 0
    result_ids = ids[0][valid]
    result_scores = scores[0][valid]

    results = papers.iloc[result_ids].copy()
    results["score"] = result_scores

    # print(f"Vector search latency: {elapsed_ms:.2f} ms")
    return results


if __name__ == "__main__":
    if not FAISS_PATH.exists() or not EMBEDDINGS_PATH.exists():
        build_index()

    query = input("Enter a research query: ")
    results = dense_search(query, k=10)

    for rank, (_, paper) in enumerate(
        results.iterrows(), start=1
    ):
        print(f"\n{rank}. {paper['title']}")
        print(f"   Similarity: {paper['score']:.4f}")
        print(f"   Abstract: {paper['abstract'][:350]}...")
        print(f"   Link: {paper['url']}")
