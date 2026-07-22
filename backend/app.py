import json
import os
import time
import re
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS

# Compatibility shim: some transformers/sentence-transformers versions
# expect `is_offline_mode` to be exported from huggingface_hub. If it's
# missing in the installed package, inject a simple fallback to avoid
# import-time errors when importing `sentence_transformers`.
try:
    import huggingface_hub
    if not hasattr(huggingface_hub, "is_offline_mode"):
        def _hf_is_offline_mode():
            return False

        setattr(huggingface_hub, "is_offline_mode", _hf_is_offline_mode)
except Exception:
    # If huggingface_hub isn't installed yet, let later imports raise a clear error
    pass

from backend.bm25 import BM25Search
from backend.faiss_search import FAISSSearch
from backend.hybrid import HybridSearch
from backend.rerank import RerankSearch
from backend.rag import RAGSearch
from deep_translator import GoogleTranslator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET = "scifact"
AVAILABLE_DATASETS = ["scifact", "nfcorpus", "arguana", "fiqa"]
SAMPLE_SIZE = None
ALPHA = 0.5
TOP_K_DEFAULT = 10

app = Flask(__name__)
CORS(app)

print("=" * 50)
print("🚀 INITIALISATION DU MOTEUR DE RECHERCHE")
print("=" * 50)
print(f"📚 Dataset : {DATASET}")
print(f"📄 Sample size : {SAMPLE_SIZE if SAMPLE_SIZE else 'Tout le corpus'}")
print(f"⚖️ Alpha hybride : {ALPHA}")
print("=" * 50)

engines = {}


def get_engine(name, dataset_name=None):
    dataset_name = dataset_name or DATASET
    cache_key = f"{name}:{dataset_name}"
    if cache_key in engines:
        return engines[cache_key]

    if name == "bm25":
        engines[cache_key] = BM25Search(dataset_name, sample_size=SAMPLE_SIZE)
    elif name == "faiss":
        engines[cache_key] = FAISSSearch(
            model_name="distiluse-base-multilingual-cased-v2",
            dataset_name=dataset_name,
        )
    elif name == "hybrid":
        engines[cache_key] = HybridSearch(dataset_name, sample_size=SAMPLE_SIZE, alpha=ALPHA)
    elif name == "rerank":
        engines[cache_key] = RerankSearch(dataset_name, sample_size=SAMPLE_SIZE)
    elif name == "rag":
        engines[cache_key] = RAGSearch(dataset_name, sample_size=SAMPLE_SIZE)
    else:
        raise ValueError(f"Moteur inconnu : {name}")

    return engines[cache_key]

print("✅ API PRÊTE - En attente des requêtes...")
print("=" * 50)

try:
    translator = GoogleTranslator(source="auto", target="en")
    USE_DEEP_TRANSLATOR = True
    print("✅ deep-translator chargé")
except Exception as exc:
    translator = None
    USE_DEEP_TRANSLATOR = False
    print(f"⚠️ deep-translator non disponible : {exc}")

FALLBACK_TRANSLATIONS = {
    "é": "e",
    "è": "e",
    "ê": "e",
    "à": "a",
    "â": "a",
    "ç": "c",
    "œ": "oe",
    "æ": "ae",
}


def translate_to_en(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    if USE_DEEP_TRANSLATOR and translator:
        try:
            return translator.translate(text)
        except Exception as exc:
            logger.warning(f"deep-translator erreur: {exc}")
    lower_text = text.lower()
    for fr, en in FALLBACK_TRANSLATIONS.items():
        lower_text = lower_text.replace(fr, en)
    return lower_text


def detect_language(text: str) -> str:
    if not text or not isinstance(text, str):
        return "en"
    return "fr" if re.search(r"[éèêëàâäîïôöûüçœæ]", text.lower()) else "en"


def prepare_query(query: str):
    if detect_language(query) == "fr":
        return query, translate_to_en(query)
    return query, query


def process_search(engine_name: str, query: str, top_k: int = TOP_K_DEFAULT, dataset_name: str = None):
    query_original, query_used = prepare_query(query)
    logger.info(f"🔍 {engine_name.upper()} - dataset: {dataset_name or DATASET} - query_original: {query_original}")
    logger.info(f"🔍 {engine_name.upper()} - query_used: {query_used}")

    engine = get_engine(engine_name, dataset_name=dataset_name)
    start_time = time.time()
    raw_results = engine.search(query_used, top_k=top_k)
    elapsed_ms = round((time.time() - start_time) * 1000, 2)

    if isinstance(raw_results, dict):
        return {
            "method": engine_name.upper(),
            "query_original": query_original,
            "query_used": query_used,
            "time_ms": elapsed_ms,
            "response": raw_results.get("response", ""),
            "sources": raw_results.get("sources", []),
        }

    return {
        "method": engine_name.upper(),
        "query_original": query_original,
        "query_used": query_used,
        "total_results": len(raw_results),
        "time_ms": elapsed_ms,
        "results": raw_results,
    }


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Bienvenue sur l'API RAG - Moteur de recherche intelligent",
        "endpoints": {
            "POST /search_bm25": "BM25 - Recherche lexicale",
            "POST /search_faiss": "FAISS - Recherche sémantique",
            "POST /search_hybrid": "Hybride - BM25 + FAISS",
            "POST /search_rerank": "Reranking - Cross-Encoder",
            "POST /search_rag": "RAG - Génération de réponse",
        },
        "configuration": {
            "default_dataset": DATASET,
            "available_datasets": AVAILABLE_DATASETS,
            "sample_size": SAMPLE_SIZE,
            "alpha": ALPHA,
        },
        "status": "ready",
        "translation_mode": "deep-translator" if USE_DEEP_TRANSLATOR else "fallback dictionnaire",
    })


def parse_request_json():
    data = request.get_json(silent=True)
    if data is None:
        try:
            raw_body = request.get_data(as_text=True)
            if raw_body:
                data = json.loads(raw_body)
        except Exception as exc:
            logger.warning(f"JSON parse fallback failed: {exc}")
            data = None
    return data


def validate_request(data):
    if not data or not isinstance(data, dict) or "query" not in data:
        return False, "Requête JSON invalide. Utilisez {'query': 'texte', 'top_k': 10, 'dataset': 'scifact'}."
    if not isinstance(data["query"], str) or not data["query"].strip():
        return False, "Le champ 'query' doit être une chaîne non vide."
    dataset_name = data.get("dataset", DATASET)
    if dataset_name not in AVAILABLE_DATASETS:
        return False, f"Dataset invalide. Choisissez l'un des datasets suivants : {AVAILABLE_DATASETS}."
    return True, None


@app.route("/search_bm25", methods=["POST"])
def search_bm25():
    data = parse_request_json()
    valid, error = validate_request(data)
    if not valid:
        return jsonify({"error": error}), 400
    try:
        dataset_name = data.get("dataset", DATASET)
        return jsonify(process_search("bm25", data["query"].strip(), int(data.get("top_k", TOP_K_DEFAULT)), dataset_name=dataset_name))
    except Exception:
        logger.exception("Erreur BM25")
        return jsonify({"error": "Erreur interne BM25."}), 500


@app.route("/search_faiss", methods=["POST"])
def search_faiss():
    data = parse_request_json()
    valid, error = validate_request(data)
    if not valid:
        return jsonify({"error": error}), 400
    try:
        dataset_name = data.get("dataset", DATASET)
        return jsonify(process_search("faiss", data["query"].strip(), int(data.get("top_k", TOP_K_DEFAULT)), dataset_name=dataset_name))
    except FileNotFoundError as exc:
        logger.error(exc)
        return jsonify({"error": str(exc)}), 500
    except Exception:
        logger.exception("Erreur FAISS")
        return jsonify({"error": "Erreur interne FAISS."}), 500


@app.route("/search_hybrid", methods=["POST"])
def search_hybrid():
    data = parse_request_json()
    valid, error = validate_request(data)
    if not valid:
        return jsonify({"error": error}), 400
    try:
        dataset_name = data.get("dataset", DATASET)
        return jsonify(process_search("hybrid", data["query"].strip(), int(data.get("top_k", TOP_K_DEFAULT)), dataset_name=dataset_name))
    except Exception:
        logger.exception("Erreur Hybride")
        return jsonify({"error": "Erreur interne Hybride."}), 500


@app.route("/search_rerank", methods=["POST"])
def search_rerank():
    data = parse_request_json()
    valid, error = validate_request(data)
    if not valid:
        return jsonify({"error": error}), 400
    try:
        dataset_name = data.get("dataset", DATASET)
        return jsonify(process_search("rerank", data["query"].strip(), int(data.get("top_k", TOP_K_DEFAULT)), dataset_name=dataset_name))
    except Exception:
        logger.exception("Erreur Rerank")
        return jsonify({"error": "Erreur interne Rerank."}), 500


@app.route("/search_rag", methods=["POST"])
def search_rag():
    data = parse_request_json()
    valid, error = validate_request(data)
    if not valid:
        return jsonify({"error": error}), 400
    try:
        dataset_name = data.get("dataset", DATASET)
        return jsonify(process_search("rag", data["query"].strip(), int(data.get("top_k", 3)), dataset_name=dataset_name))
    except Exception:
        logger.exception("Erreur RAG")
        return jsonify({"error": "Erreur interne RAG."}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint non trouvé."}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Erreur interne du serveur."}), 500


if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "0").lower() in {"1", "true", "yes", "on"}
    app.run(host="0.0.0.0", port=5000, debug=debug_mode, use_reloader=debug_mode)
