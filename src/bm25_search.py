
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

DATA_PATH = Path("data/papers.csv")


def tokenize(text):
    """Convert text into lowercase word tokens."""
    return re.findall(r"\b\w+\b", str(text).lower())


def load_papers():
    papers = pd.read_csv(DATA_PATH).fillna("")
    papers = papers.drop_duplicates("paper_id")
    papers = papers.reset_index(drop=True)

    papers["search_text"] = (
        papers["title"].astype(str)
        + ". "
        + papers["abstract"].astype(str)
    )

    return papers


papers = load_papers()

# Build the lexical index once when the module loads.
tokenized_documents = [
    tokenize(text) for text in papers["search_text"]
]

bm25 = BM25Okapi(tokenized_documents)


def search(query, k=10):
    if not query.strip():
        return papers.iloc[0:0].copy()

    start_time = time.perf_counter()

    scores = bm25.get_scores(tokenize(query))
    top_ids = np.argsort(scores)[::-1][:k]

    results = papers.iloc[top_ids].copy()
    results["score"] = scores[top_ids]

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    # print(f"Search latency: {elapsed_ms:.2f} ms")
    return results


if __name__ == "__main__":
    query = input("Enter a research query: ")

    results = search(query, k=10)

    for rank, (_, paper) in enumerate(
        results.iterrows(), start=1
    ):
        print(f"\n{rank}. {paper['title']}")
        print(f"   Score: {paper['score']:.4f}")
        print(f"   Authors: {paper['authors']}")
        print(f"   Abstract: {paper['abstract'][:350]}...")
        print(f"   Link: {paper['url']}")
