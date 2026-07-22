import re
from backend.rerank import RerankSearch


class RAGSearch:
    def __init__(self, dataset_name="scifact", sample_size=None):
        print("🚀 Initialisation du moteur RAG...")
        self.rerank = RerankSearch(dataset_name, sample_size)

    def search(self, query, top_k=3):
        docs = self.rerank.search(query, top_k=5)
        if not docs:
            return {"response": "", "sources": []}

        response = self._generate_answer(query, docs)
        return {
            "response": response,
            "sources": docs,
        }

    def _generate_answer(self, query: str, docs: list[dict]) -> str:
        query_terms = set(re.findall(r"\w+", query.lower()))
        sentences = []
        for doc in docs:
            for candidate in re.split(r"(?<=[.!?])\s+", doc["text"]):
                cleaned = candidate.strip()
                if not cleaned:
                    continue
                words = set(re.findall(r"\w+", cleaned.lower()))
                if len(words & query_terms) >= 2:
                    sentences.append(cleaned)

        if sentences:
            return " ".join(sentences[:3])

        first_doc_text = docs[0]["text"]
        return first_doc_text[:400] + ("..." if len(first_doc_text) > 400 else "")
