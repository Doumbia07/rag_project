"""
API Flask pour le moteur de recherche RAG.
Expose les endpoints /search_bm25, /search_faiss, etc.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from backend.bm25 import BM25Search
import time

# Initialiser l'application Flask
app = Flask(__name__)
CORS(app)  # Autoriser les requêtes depuis le frontend (Streamlit ou Base44)

print("=" * 50)
print("🚀 INITIALISATION DU MOTEUR DE RECHERCHE")
print("=" * 50)

# Initialiser le moteur BM25 (au démarrage de l'API)
# Utilise nfcorpus (petit) pour les tests rapides, puis tu pourras passer à scifact
bm25_engine = BM25Search(dataset_name="nfcorpus", sample_size=500)

print("=" * 50)
print("✅ API PRÊTE - En attente des requêtes...")
print("=" * 50)


@app.route("/", methods=["GET"])
def home():
    """Page d'accueil de l'API."""
    return jsonify({
        "message": "Bienvenue sur l'API RAG - Moteur de recherche intelligent",
        "endpoints": {
            "POST /search_bm25": "Recherche BM25 (baseline)",
            "POST /search_faiss": "Recherche par embeddings (bientôt)",
            "POST /search_hybrid": "Recherche hybride (bientôt)",
            "POST /search_rerank": "Reranking (bientôt)"
        },
        "dataset": "nfcorpus (échantillon de 500 documents)",
        "status": "ready"
    })


@app.route("/search_bm25", methods=["POST"])
def search_bm25():
    """
    Endpoint de recherche BM25.
    Body: {"query": "votre question", "top_k": 10}
    """
    # Récupérer les données de la requête
    data = request.get_json()
    
    if not data or "query" not in data:
        return jsonify({"error": "Requête invalide. Envoyez {'query': 'votre question'}"}), 400
    
    query = data["query"].strip()
    top_k = data.get("top_k", 10)  # Par défaut 10 résultats
    
    if not query:
        return jsonify({"error": "La requête ne peut pas être vide"}), 400
    
    # Mesurer le temps de recherche
    start_time = time.time()
    results = bm25_engine.search(query, top_k=top_k)
    elapsed_time = time.time() - start_time
    
    # Construire la réponse
    response = {
        "method": "BM25",
        "query": query,
        "total_results": len(results),
        "time_ms": round(elapsed_time * 1000, 2),
        "results": results
    }
    
    return jsonify(response)


@app.route("/search_faiss", methods=["POST"])
def search_faiss():
    """Placeholder pour la recherche FAISS (à implémenter)."""
    return jsonify({
        "method": "FAISS",
        "message": "Endpoint en cours de développement. Revenez bientôt !"
    })


@app.route("/search_hybrid", methods=["POST"])
def search_hybrid():
    """Placeholder pour la recherche hybride (à implémenter)."""
    return jsonify({
        "method": "Hybride",
        "message": "Endpoint en cours de développement. Revenez bientôt !"
    })


@app.route("/search_rerank", methods=["POST"])
def search_rerank():
    """Placeholder pour le reranking (à implémenter)."""
    return jsonify({
        "method": "Reranking",
        "message": "Endpoint en cours de développement. Revenez bientôt !"
    })


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint non trouvé"}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Erreur interne du serveur"}), 500


if __name__ == "__main__":
    # Lancer l'API sur le port 5000
    app.run(host="0.0.0.0", port=5000, debug=True)