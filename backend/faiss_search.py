import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class FAISSSearch:
    def __init__(
        self,
        model_name="distiluse-base-multilingual-cased-v2",
        index_path="models/faiss_index.index",
        metadata_path="models/metadata.pkl",
    ):
        print("🚀 Initialisation FAISS pré-généré...")
        self.model_name = model_name
        self.index_path = os.path.abspath(index_path)
        self.metadata_path = os.path.abspath(metadata_path)

        if not os.path.exists(self.index_path):
            raise FileNotFoundError(
                f"Fichier FAISS manquant : {self.index_path}. Placez faiss_index.index dans models/."
            )
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(
                f"Fichier metadata manquant : {self.metadata_path}. Placez metadata.pkl dans models/."
            )

        print(f"📂 Chargement de l'index FAISS depuis {self.index_path}")
        self.index = faiss.read_index(self.index_path)

        print(f"📂 Chargement des métadonnées depuis {self.metadata_path}")
        self.doc_ids, self.doc_titles, self.doc_texts = self._load_metadata(self.metadata_path)

        print(f"📄 {len(self.doc_ids)} documents FAISS chargés.")
        print(f"🔤 Chargement du modèle d'embeddings {self.model_name}...")
        self.model = SentenceTransformer(self.model_name)

    def _load_metadata(self, metadata_path):
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)

        if isinstance(metadata, dict):
            # Common variant: keys named 'doc_ids', 'doc_titles', 'doc_texts'
            if all(key in metadata for key in ["doc_ids", "doc_titles", "doc_texts"]):
                return metadata["doc_ids"], metadata["doc_titles"], metadata["doc_texts"]

            # Variant expected by older code: 'doc_ids', 'titles', 'texts'
            if all(key in metadata for key in ["doc_ids", "titles", "texts"]):
                return metadata["doc_ids"], metadata["titles"], metadata["texts"]

            # If metadata is a mapping of doc_id -> {title,text}
            if all(isinstance(value, dict) for value in metadata.values()):
                doc_ids = list(metadata.keys())
                titles = [metadata[doc_id].get("title", "") for doc_id in doc_ids]
                texts = [metadata[doc_id].get("text", "") for doc_id in doc_ids]
                return doc_ids, titles, texts

            # If metadata contains a 'corpus' dict
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

    def search(self, query, top_k=10):
        if not query or not isinstance(query, str):
            return []

        query_embedding = self.model.encode([query], convert_to_numpy=True)
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
