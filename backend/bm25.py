"""
Moteur de recherche BM25 (baseline).
Utilise rank_bm25 pour l'indexation et la recherche.
"""

from rank_bm25 import BM25Okapi
import re
from backend.data_loader import load_dataset

class BM25Search:
    def __init__(self, dataset_name="nfcorpus", sample_size=None):
        """
        Initialise le moteur BM25 avec un dataset BEIR.
        
        Args:
            dataset_name (str): "nfcorpus" ou "scifact"
            sample_size (int): Nombre de documents à indexer (None = tout)
        """
        print(f"🚀 Initialisation du moteur BM25 sur {dataset_name}...")
        
        # Charger les données
        self.corpus, self.queries, self.qrels = load_dataset(dataset_name, sample_size)
        
        # Préparer les documents
        self.doc_ids = list(self.corpus.keys())
        self.doc_texts = []
        self.doc_titles = []
        
        for doc_id in self.doc_ids:
            doc = self.corpus[doc_id]
            self.doc_titles.append(doc.get("title", ""))
            self.doc_texts.append(doc.get("text", ""))
        
        # Tokeniser les documents
        print("🔤 Tokenisation des documents...")
        self.tokenized_docs = [self._tokenize(text) for text in self.doc_texts]
        
        # Construire l'index BM25
        print("📊 Construction de l'index BM25...")
        self.bm25 = BM25Okapi(self.tokenized_docs)
        print(f"✅ Index BM25 construit avec {len(self.doc_ids)} documents.")
    
    def _tokenize(self, text):
        """Tokenise un texte : minuscules + uniquement lettres."""
        text = text.lower()
        return re.findall(r'\w+', text)
    
    def search(self, query, top_k=10):
        """
        Recherche les documents les plus pertinents pour une requête.
        
        Args:
            query (str): La question de l'utilisateur
            top_k (int): Nombre de résultats à retourner
        
        Returns:
            list: [{"doc_id": ..., "title": ..., "text": ..., "score": ...}]
        """
        tokens = self._tokenize(query)
        
        # Si la requête est vide
        if not tokens:
            return []
        
        # Calculer les scores BM25
        scores = self.bm25.get_scores(tokens)
        
        # Trier par score décroissant
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        # Construire les résultats
        results = []
        for idx in top_indices:
            # Ignorer les scores nuls
            if scores[idx] == 0:
                continue
                
            results.append({
                "doc_id": self.doc_ids[idx],
                "title": self.doc_titles[idx][:100] + "..." if len(self.doc_titles[idx]) > 100 else self.doc_titles[idx],
                "text": self.doc_texts[idx][:500] + "..." if len(self.doc_texts[idx]) > 500 else self.doc_texts[idx],
                "score": float(scores[idx])
            })
        
        return results


# Test rapide
if __name__ == "__main__":
    print("🔍 Test du moteur BM25...")
    searcher = BM25Search("nfcorpus", sample_size=500)
    
    test_queries = [
        "effect of caffeine on sleep",
        "what is sleep apnea",
        "treatment for insomnia"
    ]
    
    for q in test_queries:
        print(f"\n📝 Requête : {q}")
        results = searcher.search(q, top_k=3)
        for i, r in enumerate(results):
            print(f"  {i+1}. [{r['score']:.2f}] {r['title'][:60]}...")