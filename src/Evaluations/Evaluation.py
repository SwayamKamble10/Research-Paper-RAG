import pandas as pd
import numpy as np
from hybrid_search import hybrid_search

queries = [
    "How can transformer models improve long-document summarization?",
    "What methods reduce hallucinations in large language models?",
    "How does retrieval-augmented generation improve factual accuracy?",
    "What are effective ways to evaluate question-answering systems?",
    "How do instruction-tuning and preference optimization affect language-model behavior?",
    "What techniques improve multilingual information retrieval?",
    "How can small language models perform well with limited compute?",
    "What are the limitations of using synthetic data to train language models?",
    "How can models detect and mitigate bias in generated text?",
    "What methods improve reasoning in large language models?",
    "How can graph neural networks be used for drug discovery?",
    "What machine-learning methods predict protein structure or function?",
    "How can deep learning identify rare diseases from clinical records?",
    "What are effective approaches for medical image segmentation?",
    "How can federated learning protect privacy in healthcare?",
    "How can models handle missing or noisy medical data?",
    "What methods improve interpretability of clinical prediction models?",
    "How can wearable sensor data be used for early disease detection?",
    "What are the challenges of applying machine learning to electronic health records?",
    "How can causal inference support treatment-effect estimation?",
    "How can reinforcement learning improve robotic manipulation?",
    "What methods help autonomous vehicles predict pedestrian behavior?",
    "How can robots learn tasks from human demonstrations?",
    "What are effective approaches to multi-agent reinforcement learning?",
    "How can vision-language models support embodied AI?",
    "How can machine learning forecast electricity demand?",
    "What methods detect anomalies in industrial sensor time series?",
    "How can satellite imagery support crop-yield prediction?",
    "What models improve short-term weather forecasting?",
    "How can optimization reduce emissions in transportation networks?",
    "How can graph-based methods detect financial fraud?",
    "What machine-learning approaches forecast stock-market volatility?",
    "How can natural-language processing extract events from financial news?",
    "What methods improve credit-risk prediction under class imbalance?",
    "How can explainable AI support financial decision-making?",
    "How can contrastive learning improve image representation?",
    "What are the trade-offs between vision transformers and convolutional neural networks?",
    "How can self-supervised learning reduce the need for labeled data?",
    "What methods improve domain adaptation when training and test data differ?",
    "How can uncertainty estimation make neural-network predictions more reliable?",
    "How can differential privacy be applied to machine-learning models?",
    "What methods defend neural networks against adversarial examples?",
    "How can data poisoning attacks be detected during model training?",
    "What are the security risks of deploying large language models with tools?",
    "How can machine learning systems be made more robust to distribution shift?",
    "How can vector databases improve semantic search over scientific papers?",
    "How does reciprocal rank fusion compare with other hybrid retrieval methods?",
    "What methods improve citation recommendation for scientific articles?",
    "How can scientific-paper abstracts be used to predict research impact?",
    "What approaches evaluate the quality of retrieved documents in a research-paper search system?"
]


TOP_K = 10
RESULTS_FILE = "retrieval_results.csv"
JUDGMENTS_FILE = "human_relevance_judgments.csv"
METRICS_FILE = "retrieval_metrics.csv"


# ---------- Pass 1: Run retrieval and save results ----------

rows = []

for query in queries:
    result = hybrid_search(query=query, top_k=TOP_K)

    # Your print output shows result.paper_id is a pandas Series.
    paper_ids = result.paper_id
    if hasattr(paper_ids, "tolist"):
        paper_ids = paper_ids.tolist()
    elif isinstance(paper_ids, str):
        paper_ids = [paper_ids]
    else:
        paper_ids = list(paper_ids)

    for rank, paper_id in enumerate(paper_ids[:TOP_K], start=1):
        rows.append({
            "query": query,
            "rank": rank,
            "paper_id": str(paper_id),
            "relevance_grade": ""  # Fill this in after reviewing the paper.
        })

retrieval_df = pd.DataFrame(rows)
retrieval_df.to_csv(RESULTS_FILE, index=False)

print(f"Saved ranked results to {RESULTS_FILE}")
print("Review each paper's title/abstract and enter a relevance grade:")
print("3 = highly relevant, 2 = relevant, 1 = partially relevant, 0 = not relevant")


# ---------- Pass 2: After filling in relevance_grade ----------

# Save your reviewed file as human_judgments.csv, then run this section.
# The relevance_grade column must contain values from 0 to 3.

def reciprocal_rank(grades):
    """Reciprocal rank of the first relevant result (grade >= 2)."""
    for rank, grade in enumerate(grades, start=1):
        if grade >= 2:
            return 1 / rank
    return 0.0


def ndcg_at_k(grades, k=10):
    """nDCG using graded relevance and gain = 2^grade - 1."""
    grades = np.asarray(grades[:k], dtype=float)

    discounts = np.log2(np.arange(2, len(grades) + 2))
    gains = (2 ** grades) - 1
    dcg = np.sum(gains / discounts)

    ideal_grades = np.sort(grades)[::-1]
    ideal_gains = (2 ** ideal_grades) - 1
    idcg = np.sum(ideal_gains / discounts)

    return float(dcg / idcg) if idcg > 0 else 0.0


judgments = pd.read_csv(JUDGMENTS_FILE)

# Validate that every result has a valid human judgment.
judgments["relevance_grade"] = pd.to_numeric(
    judgments["relevance_grade"], errors="coerce"
)

if judgments["relevance_grade"].isna().any():
    raise ValueError(
        f"Some relevance grades are missing. Fill them in {JUDGMENTS_FILE} first."
    )

if not judgments["relevance_grade"].between(0, 3).all():
    raise ValueError("Relevance grades must be integers from 0 to 3.")

judgments["relevance_grade"] = judgments["relevance_grade"].astype(int)

# Sort each query's results by their original retrieval rank.
judgments = judgments.sort_values(["query", "rank"])

metrics = []

for query, group in judgments.groupby("query", sort=False):
    grades = group["relevance_grade"].tolist()
    relevant_count = sum(grade >= 2 for grade in grades)

    metrics.append({
        "query": query,
        "num_judged": len(grades),
        "relevant_in_top_k": relevant_count,
        "precision_at_k": relevant_count / len(grades) if grades else 0.0,
        "mrr_at_k": reciprocal_rank(grades),
        "ndcg_at_k": ndcg_at_k(grades, k=TOP_K),
    })

metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv(METRICS_FILE, index=False)

print("\nPer-query metrics:")
print(metrics_df.to_string(index=False))

print("\nMean metrics across queries:")
print(
    metrics_df[
        ["precision_at_k", "mrr_at_k", "ndcg_at_k"]
    ].mean().round(4)
)

print(f"\nSaved per-query metrics to {METRICS_FILE}")