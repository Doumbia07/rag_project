import re
from backend.rerank import RerankSearch

# Tentative d'import du LLM (optionnel)
try:
    from backend.llm import MistralLLM
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    MistralLLM = None


class RAGSearch:
    def __init__(self, dataset_name="scifact", sample_size=None, use_llm=True):
        """
        Initialise le moteur RAG.
        Args:
            dataset_name: Nom du dataset
            sample_size: Taille de l'échantillon
            use_llm: Activer/désactiver l'utilisation du LLM (Mistral)
        """
        print("🚀 Initialisation du moteur RAG...")
        self.rerank = RerankSearch(dataset_name, sample_size)
        self.use_llm = use_llm and LLM_AVAILABLE

        if self.use_llm:
            try:
                self.llm = MistralLLM()
                print("✅ LLM (Mistral) chargé avec succès")
            except Exception as e:
                print(f"⚠️ Erreur de chargement du LLM : {e}")
                self.use_llm = False
        else:
            self.llm = None
            if use_llm and not LLM_AVAILABLE:
                print("⚠️ LLM non disponible (module backend.llm introuvable)")

    def search(self, query, top_k=3):
        docs = self.rerank.search(query, top_k=5)
        if not docs:
            return {"response": "", "sources": []}

        # Si le LLM est disponible et activé, on l'utilise
        if self.use_llm and self.llm:
            response = self.llm.generate_response(query, docs)
        else:
            # Sinon, on utilise la méthode de génération basique (fallback)
            response = self._generate_answer(query, docs)

        return {
            "response": response,
            "sources": docs,
        }

    def _generate_answer(self, query: str, docs: list[dict]) -> str:
        """Méthode de génération basique (fallback)"""
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