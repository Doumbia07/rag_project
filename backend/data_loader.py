"""
Charge les données BEIR (SciFact ou NFCorpus)
"""

from beir import util
from beir.datasets.data_loader import GenericDataLoader
import os
import itertools

def load_dataset(dataset_name="nfcorpus", sample_size=None):
    """
    Télécharge et charge un dataset BEIR.
    
    Args:
        dataset_name (str): "nfcorpus" (recommandé pour tests) ou "scifact"
        sample_size (int): Nombre de documents à charger (None = tout)
    
    Returns:
        corpus (dict): {doc_id: {"title": ..., "text": ...}}
        queries (dict): {query_id: "question"}
        qrels (dict): {query_id: {doc_id: relevance_score}}
    """
    
    # Chemin où les données seront stockées
    data_path = f"data/{dataset_name}"
    
    # Télécharger si le dossier n'existe pas
    if not os.path.exists(data_path):
        print(f"📥 Téléchargement de {dataset_name}...")
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
        util.download_and_unzip(url, "data")
        print("✅ Téléchargement terminé.")
    
    # Charger les données
    print(f"📂 Chargement de {dataset_name}...")
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    
    # Si on veut un échantillon plus petit (pour les tests rapides)
    if sample_size:
        print(f"📊 Réduction à {sample_size} documents et 50 requêtes...")
        limited_corpus = dict(itertools.islice(corpus.items(), sample_size))
        limited_queries = dict(itertools.islice(queries.items(), 50))
        return limited_corpus, limited_queries, qrels
    
    print(f"✅ Chargé : {len(corpus)} documents, {len(queries)} requêtes")
    return corpus, queries, qrels


# Test rapide si le fichier est exécuté directement
if __name__ == "__main__":
    print("🔍 Test du chargeur de données...")
    corpus, queries, qrels = load_dataset("nfcorpus", sample_size=500)
    print(f"✅ Corpus : {len(corpus)} documents")
    print(f"✅ Requêtes : {len(queries)}")
    print(f"📝 Exemple de requête : {list(queries.values())[0][:100]}...")