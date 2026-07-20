import os
import itertools
import shutil
import zipfile
from beir import util
from beir.datasets.data_loader import GenericDataLoader

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")


def _ensure_dataset(dataset_name: str) -> str:
    dataset_path = os.path.join(DATA_DIR, dataset_name)
    zip_path = os.path.join(DATA_DIR, f"{dataset_name}.zip")

    if os.path.exists(zip_path) and not zipfile.is_zipfile(zip_path):
        print(f"⚠️ Fichier ZIP corrompu détecté : {zip_path}. Suppression en cours...")
        os.remove(zip_path)

    if os.path.exists(dataset_path):
        expected_corpus = os.path.join(dataset_path, "corpus.jsonl")
        if not os.path.exists(expected_corpus):
            print(f"⚠️ Le dossier du dataset est incomplet : {dataset_path}. Suppression en cours...")
            shutil.rmtree(dataset_path, ignore_errors=True)

    if not os.path.exists(dataset_path):
        print(f"📥 Dataset '{dataset_name}' absent. Téléchargement BEIR en cours...")
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
        util.download_and_unzip(url, DATA_DIR)
        print("✅ Téléchargement BEIR terminé.")
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Le dataset {dataset_name} n'a pas pu être chargé après téléchargement.")
    return dataset_path


def load_dataset(dataset_name="scifact", sample_size=None):
    """Charge un dataset BEIR et retourne (corpus, queries, qrels)."""
    data_path = _ensure_dataset(dataset_name)
    print(f"📂 Chargement du dataset BEIR '{dataset_name}' depuis {data_path}")
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

    if sample_size and isinstance(sample_size, int):
        print(f"📊 Réduction à {sample_size} documents pour accélérer le développement.")
        limited_corpus = dict(itertools.islice(corpus.items(), sample_size))
        limited_queries = dict(itertools.islice(queries.items(), min(sample_size, len(queries))))
        return limited_corpus, limited_queries, qrels

    return corpus, queries, qrels
