from sentence_transformers import CrossEncoder
from backend.hybrid import HybridSearch


class RerankSearch:
    def __init__(self, dataset_name="scifact", sample_size=None):
        print("🚀 Initialisation du moteur Reranking...")
        self.hybrid = HybridSearch(dataset_name, sample_size)
        print("🔤 Chargement du Cross-Encoder ms-marco-MiniLM-L-6-v2...")
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def search(self, query, top_k=10):
        hybrid_results = self.hybrid.search(query, top_k=50)

        if not hybrid_results:
            return []

        pairs = [[query, doc["text"]] for doc in hybrid_results]
        scores = self.cross_encoder.predict(pairs)

        for doc, score in zip(hybrid_results, scores):
            doc["score"] = float(score)

        reranked = sorted(hybrid_results, key=lambda x: x["score"], reverse=True)[:top_k]
        return reranked
