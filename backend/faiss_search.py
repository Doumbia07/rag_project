import os
import pickle
import faiss
import numpy as np
import hashlib  # 🔥 AJOUTÉ
from sentence_transformers import SentenceTransformer


class FAISSLoadError(RuntimeError):
    pass


def resolve_faiss_paths(dataset_name="scifact", base_dir=None):
    if base_dir is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    models_dir = os.path.join(base_dir, "models")
    normalized_name = (dataset_name or "").strip().lower()

    candidates = []
    if normalized_name:
        candidates.append((
            os.path.join(models_dir, f"{normalized_name}_index.index"),
            os.path.join(models_dir, f"{normalized_name}_metadata.pkl"),
        ))

    candidates.append((
        os.path.join(models_dir, "faiss_index.index"),
        os.path.join(models_dir, "metadata.pkl"),
    ))

    for index_path, metadata_path in candidates:
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            return index_path, metadata_path

    if normalized_name:
        return (
            os.path.join(models_dir, f"{normalized_name}_index.index"),
            os.path.join(models_dir, f"{normalized_name}_metadata.pkl"),
        )

    return os.path.join(models_dir, "faiss_index.index"), os.path.join(models_dir, "metadata.pkl")


class FAISSSearch:
    def __init__(
        self,
        model_name="distiluse-base-multilingual-cased-v2",
        index_path=None,
        metadata_path=None,
        dataset_name="scifact",
        base_dir=None,
        use_cache=True,      # 🔥 NOUVEAU : activer/désactiver le cache
        cache_size=1000,     # 🔥 NOUVEAU : taille max du cache
    ):
        print("🚀 Initialisation FAISS pré-généré...")
        self.model_name = model_name
        self.dataset_name = dataset_name or "scifact"
        self.use_cache = use_cache          # 🔥 NOUVEAU
        self.cache_size = cache_size        # 🔥 NOUVEAU
        self.cache = {}                     # 🔥 NOUVEAU
        self.cache_hits = 0                 # 🔥 NOUVEAU
        self.cache_misses = 0               # 🔥 NOUVEAU

        resolved_index_path, resolved_metadata_path = resolve_faiss_paths(
            self.dataset_name,
            base_dir=base_dir,
        )
        if index_path is not None:
            resolved_index_path = index_path
        if metadata_path is not None:
            resolved_metadata_path = metadata_path

        self.index_path = os.path.abspath(resolved_index_path)
        self.metadata_path = os.path.abspath(resolved_metadata_path)

        if not os.path.exists(self.index_path):
            raise FileNotFoundError(
                f"Fichier FAISS manquant : {self.index_path}. Placez le fichier d'index correspondant dans models/."
            )
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(
                f"Fichier metadata manquant : {self.metadata_path}. Placez le fichier de métadonnées correspondant dans models/."
            )

        print(f"📂 Chargement de l'index FAISS depuis {self.index_path}")
        self.index = faiss.read_index(self.index_path)

        print(f"📂 Chargement des métadonnées depuis {self.metadata_path}")
        self.doc_ids, self.doc_titles, self.doc_texts = self._load_metadata(self.metadata_path)

        print(f"📄 {len(self.doc_ids)} documents FAISS chargés.")
        print(f"🔤 Chargement du modèle d'embeddings {self.model_name}...")
        try:
            self.model = SentenceTransformer(self.model_name)
        except Exception as exc:
            self.model = None
            raise FAISSLoadError(f"Échec du chargement du modèle d'embeddings FAISS : {exc}") from exc

        # 🔥 INFO : état du cache
        if self.use_cache:
            print(f"⚡ Cache activé (taille max : {self.cache_size})")
        else:
            print("⚡ Cache désactivé")

    def _load_metadata(self, metadata_path):
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)

        if isinstance(metadata, dict):
            if all(key in metadata for key in ["doc_ids", "doc_titles", "doc_texts"]):
                return metadata["doc_ids"], metadata["doc_titles"], metadata["doc_texts"]

            if all(key in metadata for key in ["doc_ids", "titles", "texts"]):
                return metadata["doc_ids"], metadata["titles"], metadata["texts"]

            if all(isinstance(value, dict) for value in metadata.values()):
                doc_ids = list(metadata.keys())
                titles = [metadata[doc_id].get("title", "") for doc_id in doc_ids]
                texts = [metadata[doc_id].get("text", "") for doc_id in doc_ids]
                return doc_ids, titles, texts

            if "corpus" in metadata and isinstance(metadata["corpus"], dict):
                corpus = metadata["corpus"]
                doc_ids = list(corpus.keys())
                titles = [corpus[doc_id].get("title", "") for doc_id in doc_ids]
                texts = [corpus[doc_id].get("text", "") for doc_id in doc_ids]
                return doc_ids, titles, texts

        if isinstance(metadata, list):
            doc_ids = [item.get("doc_id") or item.get("id") for item in metadata]
            titles = [item.get("title", "") for item in metadata]
            texts = [item.get("text", "") for item in metadata]
            return doc_ids, titles, texts

        raise ValueError(
            "Impossible de charger les métadonnées FAISS. Le fichier metadata.pkl doit contenir doc_ids, titles et texts."
        )

    # ============================================================
    # 🔥 NOUVEAUTES : Cache des embeddings
    # ============================================================

    def _get_cache_key(self, query: str) -> str:
        """Génère une clé de cache unique pour la requête."""
        normalized = query.lower().strip()
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()

    def _get_embedding(self, query: str):
        """
        Retourne l'embedding de la requête (avec cache si activé).
        """
        if not self.use_cache:
            return self.model.encode([query], convert_to_numpy=True)

        cache_key = self._get_cache_key(query)

        if cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key]

        # Cache miss
        self.cache_misses += 1
        embedding = self.model.encode([query], convert_to_numpy=True)

        # Ajouter au cache
        self.cache[cache_key] = embedding

        # Gérer la taille du cache (FIFO)
        if len(self.cache) > self.cache_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

        return embedding

    def get_cache_stats(self) -> dict:
        """Retourne les statistiques du cache."""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0.0
        return {
            "cache_size": len(self.cache),
            "max_cache_size": self.cache_size,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "cache_enabled": self.use_cache,
        }

    def clear_cache(self):
        """Vide le cache des embeddings."""
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        print("🗑️ Cache FAISS vidé avec succès.")

    # ============================================================
    # Méthode search modifiée pour utiliser le cache
    # ============================================================

    def search(self, query, top_k=10):
        if not query or not isinstance(query, str):
            return []
        if self.model is None:
            raise FAISSLoadError("Le modèle d'embeddings FAISS n'a pas pu être chargé.")

        # 🔥 UTILISATION DU CACHE
        query_embedding = self._get_embedding(query)

        faiss.normalize_L2(query_embedding)
        distances, indices = self.index.search(query_embedding.astype(np.float32), top_k)

        results = []
        for score, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.doc_ids):
                continue
            results.append({
                "doc_id": self.doc_ids[idx],
                "title": self.doc_titles[idx],
                "text": self.doc_texts[idx],
                "score": float(score),
            })

        return results