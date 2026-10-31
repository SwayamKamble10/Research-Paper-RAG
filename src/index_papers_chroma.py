from pathlib import Path

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer


DATA_PATH = Path("data/papers.csv")
DB_PATH = "./chroma_db"
COLLECTION_NAME = "research_papers_v1"
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

BATCH_SIZE = 128

papers = pd.read_csv(DATA_PATH).fillna("")
papers = papers.drop_duplicates("paper_id").reset_index(drop=True)

papers["search_text"] = (
    papers["title"].astype(str)
    + ". "
    + papers["abstract"].astype(str)
)
client = chromadb.PersistentClient(path=DB_PATH)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)

model = SentenceTransformer(MODEL_NAME)

def build_index():
    total = len(papers)

    for start in range(0, total, BATCH_SIZE):
        batch = papers.iloc[start:start + BATCH_SIZE]

        documents = batch["search_text"].tolist()

        embeddings = model.encode(
            documents,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).tolist()

        ids = batch["paper_id"].astype(str).tolist()

        metadatas = []
        for _, row in batch.iterrows():
            metadatas.append({
                "paper_id": str(row["paper_id"]),
                "title": str(row["title"]),
                "authors": str(row["authors"]),
                "published": str(row["published"]),
                "categories": str(row["categories"]),
                "url": str(row["url"]),
            })

        # Upsert makes repeated runs safe for existing IDs.
        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        print(f"Indexed {min(start + len(batch), total)}/{total}")

    print("Indexing complete.")
    print("Total documents in collection:", collection.count())


if __name__ == "__main__":
    build_index()
