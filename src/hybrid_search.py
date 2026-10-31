
import time
import pandas as pd

from bm25_search import search as bm25_search
from dense_search import dense_search


def reciprocal_rank_fusion(
    rankings,
    rank_constant=60,
    top_k=10,
):
    """
    Combine ranked result lists using Reciprocal Rank Fusion.

    Parameters
    ----------
    rankings : list[tuple[str, pd.DataFrame]]
        Each tuple contains a source name and a ranked DataFrame.
        Each DataFrame must contain a paper_id column.

    rank_constant : int
        RRF smoothing constant, usually 60.

    top_k : int
        Number of final results to return.

    Returns
    -------
    pd.DataFrame
        Fused results, including source ranks and RRF scores.
    """
    if rank_constant <= 0:
        raise ValueError("rank_constant must be positive.")

    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    fused_scores = {}
    source_ranks = {}
    metadata = {}

    for source_name, results in rankings:
        if results is None or results.empty:
            continue

        if "paper_id" not in results.columns:
            raise ValueError(
                f"{source_name} results must contain paper_id."
            )

        # Each source should contribute only one rank per paper.
        seen_in_source = set()

        for rank, (_, row) in enumerate(
            results.iterrows(), start=1
        ):
            paper_id = str(row["paper_id"])

            if paper_id in seen_in_source:
                continue

            seen_in_source.add(paper_id)

            fused_scores[paper_id] = (
                fused_scores.get(paper_id, 0.0)
                + 1.0 / (rank_constant + rank)
            )

            source_ranks.setdefault(paper_id, {})[
                source_name
            ] = rank

            # Preserve metadata from the first source
            # that contributes this paper.
            if paper_id not in metadata:
                metadata[paper_id] = row.to_dict()

    ranked_ids = sorted(
        fused_scores,
        key=lambda paper_id: fused_scores[paper_id],
        reverse=True,
    )[:top_k]

    output = []

    for paper_id in ranked_ids:
        row = metadata[paper_id].copy()

        row["paper_id"] = paper_id
        row["rrf_score"] = fused_scores[paper_id]
        row["bm25_rank"] = source_ranks.get(
            paper_id, {}
        ).get("BM25")

        row["dense_rank"] = source_ranks.get(
            paper_id, {}
        ).get("Dense")

        output.append(row)

    return pd.DataFrame(output)


def hybrid_search(
    query,
    top_k=10,
    candidate_k=100,
    rank_constant=60,
):
    """
    Retrieve candidates using BM25 and ChromaDB,
    then combine the rankings with RRF.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Please enter a non-empty query.")

    if top_k <= 0 or candidate_k <= 0:
        raise ValueError("top_k and candidate_k must be positive.")

    if top_k > candidate_k:
        raise ValueError(
            "top_k cannot exceed candidate_k."
        )

    start_time = time.perf_counter()

    # 1. Lexical retrieval
    bm25_results = bm25_search(
        query,
        k=candidate_k,
    )

    # 2. Dense semantic retrieval from persistent ChromaDB
    dense_results = dense_search(
        query,
        k=candidate_k,
    )

    # 3. Rank fusion
    results = reciprocal_rank_fusion(
        rankings=[
            ("BM25", bm25_results),
            ("Dense", dense_results),
        ],
        rank_constant=rank_constant,
        top_k=top_k,
    )

    elapsed_ms = (
        time.perf_counter() - start_time
    ) * 1000

    # print(f"\nHybrid search latency: {elapsed_ms:.2f} ms")
    # print(f"BM25 candidates: {len(bm25_results)}")
    # print(f"Dense candidates: {len(dense_results)}")
    # print(f"Final results: {len(results)}")

    return results


def display_results(results):
    """Display ranked papers in the terminal."""
    if results.empty:
        print("No papers found.")
        return

    for rank, (_, paper) in enumerate(
        results.iterrows(), start=1
    ):
        print(f"\n{'=' * 70}")
        print(f"Rank: {rank}")
        print(f"Title: {paper.get('title', 'Unknown')}")
        print(f"Paper ID: {paper['paper_id']}")
        print(f"RRF score: {paper['rrf_score']:.6f}")
        print(f"BM25 rank: {paper['bm25_rank']}")
        print(f"Dense rank: {paper['dense_rank']}")

        authors = paper.get("authors", "")
        if authors:
            print(f"Authors: {authors}")

        abstract = paper.get("abstract", "")
        if not abstract:
            abstract = paper.get("search_text", "")

        if abstract:
            print(f"Text: {str(abstract)[:400]}...")

        print(f"URL: {paper.get('url', '')}")


if __name__ == "__main__":
    query = input("Enter your research query: ").strip()

    try:
        results = hybrid_search(
            query=query,
            top_k=10,
            candidate_k=100,
            rank_constant=60,
        )

        display_results(results)

        # Optional: save results for later inspection.
        if not results.empty:
            results.to_csv(
                "hybrid_search_results.csv",
                index=False,
            )
            print("\nSaved results to hybrid_search_results.csv")

    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"Search error: {exc}")
