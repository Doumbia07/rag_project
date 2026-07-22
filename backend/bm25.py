import re
from rank_bm25 import BM25Okapi
from backend.data_loader import load_dataset


class BM25Search:
    def __init__(self, dataset_name="scifact", sample_size=None):
        print(f"🚀 Initialisation BM25 sur {dataset_name}...")
        self.corpus, self.queries, self.qrels = load_dataset(dataset_name, sample_size)
        self.doc_ids = list(self.corpus.keys())
        self.doc_titles = [self.corpus[doc_id].get("title", "") for doc_id in self.doc_ids]
        self.doc_texts = [self.corpus[doc_id].get("text", "") for doc_id in self.doc_ids]
        self.tokenized_docs = [self._tokenize(text) for text in self.doc_texts]
        self.bm25 = BM25Okapi(self.tokenized_docs)
        print(f"✅ Index BM25 construit avec {len(self.doc_ids)} documents.")

    def _tokenize(self, text):
        if not text or not isinstance(text, str):
            return []
        text = text.lower()
        return re.findall(r"\w+", text)

    def search(self, query, top_k=10):
        tokens = self._tokenize(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in ranked_indices:
            score = float(scores[idx])
            if score <= 0:
                continue
            results.append({
                "doc_id": self.doc_ids[idx],
                "title": self.doc_titles[idx],
                "text": self.doc_texts[idx],
                "score": score,
            })

        return results
