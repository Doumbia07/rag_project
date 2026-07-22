from backend.bm25 import BM25Search
from backend.faiss_search import FAISSSearch


class HybridSearch:
    def __init__(self, dataset_name="scifact", sample_size=None, alpha=0.5):
        print(f"🚀 Initialisation du moteur Hybride (α={alpha})...")
        self.bm25 = BM25Search(dataset_name, sample_size)
        self.faiss = FAISSSearch(
            model_name="distiluse-base-multilingual-cased-v2",
            dataset_name=dataset_name,
        )
        self.alpha = alpha

    def _rrf_score(self, rank, k=60):
        return 1.0 / (k + rank)

    def search(self, query, top_k=10):
        search_size = max(top_k * 5, 50)
        bm25_results = self.bm25.search(query, top_k=search_size)
        faiss_results = self.faiss.search(query, top_k=search_size)

        combined = {}

        for rank, doc in enumerate(bm25_results, start=1):
            doc_id = doc["doc_id"]
            combined.setdefault(doc_id, {
                "doc_id": doc_id,
                "title": doc["title"],
                "text": doc["text"],
                "score": 0.0,
            })
            combined[doc_id]["score"] += self.alpha * self._rrf_score(rank)

        for rank, doc in enumerate(faiss_results, start=1):
            doc_id = doc["doc_id"]
            combined.setdefault(doc_id, {
                "doc_id": doc_id,
                "title": doc["title"],
                "text": doc["text"],
                "score": 0.0,
            })
            combined[doc_id]["score"] += (1 - self.alpha) * self._rrf_score(rank)
            if len(doc["title"]) > len(combined[doc_id]["title"]):
                combined[doc_id]["title"] = doc["title"]
            if len(doc["text"]) > len(combined[doc_id]["text"]):
                combined[doc_id]["text"] = doc["text"]

        sorted_results = sorted(combined.values(), key=lambda item: item["score"], reverse=True)
        return sorted_results[:top_k]
