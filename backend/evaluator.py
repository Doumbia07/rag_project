import json
import math
import os
import time
import tracemalloc
from backend.bm25 import BM25Search
from backend.faiss_search import FAISSSearch
from backend.hybrid import HybridSearch
from backend.rerank import RerankSearch
from backend.rag import RAGSearch


def ndcg_at_k(relevance_scores, k=10):
    dcg = 0.0
    idcg = 0.0
    for i, rel in enumerate(relevance_scores[:k], start=1):
        dcg += (2**rel - 1) / math.log2(i + 1)
    sorted_rels = sorted(relevance_scores, reverse=True)
    for i, rel in enumerate(sorted_rels[:k], start=1):
        idcg += (2**rel - 1) / math.log2(i + 1)
    return float(dcg / idcg) if idcg > 0 else 0.0


def recall_at_k(relevant_ids, retrieved_ids, k=100):
    if not relevant_ids:
        return 0.0
    retrieved_set = set(retrieved_ids[:k])
    return float(len(relevant_ids & retrieved_set) / len(relevant_ids))


def mrr_at_k(relevant_ids, retrieved_ids, k=10):
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def evaluate_method(method, queries, qrels, top_k=10):
    metrics = {
        "queries_evaluated": 0,
        "avg_time_ms": 0.0,
        "ndcg": 0.0,
        "recall": 0.0,
        "mrr": 0.0,
    }
    total_time = 0.0
    ndcg_sum = 0.0
    recall_sum = 0.0
    mrr_sum = 0.0

    for qid, query in queries.items():
        start = time.time()
        results = method.search(query, top_k=top_k)
        elapsed = (time.time() - start) * 1000
        total_time += elapsed

        relevant_docs = set(qrels.get(qid, {}).keys())
        retrieved_ids = [doc.get("doc_id") for doc in results]
        relevance_scores = [1.0 if doc_id in relevant_docs else 0.0 for doc_id in retrieved_ids]

        ndcg_sum += ndcg_at_k(relevance_scores, top_k)
        recall_sum += recall_at_k(relevant_docs, retrieved_ids, top_k)
        mrr_sum += mrr_at_k(relevant_docs, retrieved_ids, top_k)
        metrics["queries_evaluated"] += 1

    if metrics["queries_evaluated"] > 0:
        count = metrics["queries_evaluated"]
        metrics["avg_time_ms"] = total_time / count
        metrics["ndcg"] = ndcg_sum / count
        metrics["recall"] = recall_sum / count
        metrics["mrr"] = mrr_sum / count

    return metrics


def run_evaluation(dataset_name="scifact", sample_size=None, query_limit=20):
    from backend.data_loader import load_dataset

    print("=" * 50)
    print("🔍 ÉVALUATION DES MOTEURS DE RECHERCHE")
    print("=" * 50)

    _, queries, qrels = load_dataset(dataset_name, sample_size)
    queries = dict(list(queries.items())[:query_limit])

    engines = {
        "BM25": BM25Search(dataset_name, sample_size),
        "FAISS": FAISSSearch(
            model_name="distiluse-base-multilingual-cased-v2",
            dataset_name=dataset_name,
        ),
        "Hybride": HybridSearch(dataset_name, sample_size, alpha=0.5),
        "Rerank": RerankSearch(dataset_name, sample_size),
        "RAG": RAGSearch(dataset_name, sample_size),
    }

    summary = {}
    for name, engine in engines.items():
        print(f"\n📝 Évaluation de {name}")
        tracemalloc.start()
        metrics = evaluate_method(engine, queries, qrels, top_k=10)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        metrics["memory_peak_mb"] = peak / 1024 / 1024
        summary[name] = metrics
        print(f" - avg_time_ms: {metrics['avg_time_ms']:.2f}")
        print(f" - nDCG@10: {metrics['ndcg']:.4f}")
        print(f" - Recall@10: {metrics['recall']:.4f}")
        print(f" - MRR@10: {metrics['mrr']:.4f}")
        print(f" - memory_peak_mb: {metrics['memory_peak_mb']:.2f}")

    os.makedirs("docs", exist_ok=True)
    with open("docs/evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 50)
    print("📊 Résumé comparatif")
    print("=" * 50)
    print(f"{'Méthode':<10} | {'Temps (ms)':<10} | {'nDCG@10':<8} | {'Recall@10':<10} | {'MRR@10':<8} | {'Mémoire (MB)':<12}")
    print("-" * 70)
    for name, metrics in summary.items():
        print(
            f"{name:<10} | {metrics['avg_time_ms']:<10.2f} | {metrics['ndcg']:<8.4f} | {metrics['recall']:<10.4f} | {metrics['mrr']:<8.4f} | {metrics['memory_peak_mb']:<12.2f}"
        )

    return summary


if __name__ == "__main__":
    run_evaluation()
