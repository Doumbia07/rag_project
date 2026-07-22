import os
import pickle
import faiss
import numpy as np
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

    # Fallback to the dataset-specific names even if one of the files is missing,
    # so the error message points to the expected assets.
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
    ):
        print("🚀 Initialisation FAISS pré-généré...")
        self.model_name = model_name
        self.dataset_name = dataset_name or "scifact"

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
        if self.model is None:
            raise FAISSLoadError("Le modèle d'embeddings FAISS n'a pas pu être chargé.")

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
