# 🔍 RAG Multi-Domain - Moteur de recherche intelligent

**Laboratoire de recherche sémantique multi-domaines** pour l'évaluation et la comparaison de moteurs de recherche.

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3.3-green.svg)](https://flask.palletsprojects.com/)
[![FAISS](https://img.shields.io/badge/FAISS-1.9.0-orange.svg)](https://github.com/facebookresearch/faiss)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29.0-red.svg)](https://streamlit.io/)
[![Mistral](https://img.shields.io/badge/Mistral-7B-purple.svg)](https://mistral.ai/)

---

## 📋 Table des matières

- [Présentation du projet](#-présentation-du-projet)
- [Fonctionnalités](#-fonctionnalités)
- [Architecture](#-architecture)
- [Données](#-données)
- [Améliorations récentes](#-améliorations-récentes)
- [Technologies utilisées](#-technologies-utilisées)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [API Endpoints](#-api-endpoints)
- [Évaluation](#-évaluation)
- [Déploiement](#-déploiement)
- [Documentation](#-documentation)
- [Livrables](#-livrables)
- [Auteurs](#-auteurs)
- [Licence](#-licence)

---

## 🎯 Présentation du projet

Ce projet est un **moteur de recherche intelligent** qui compare plusieurs approches de recherche d'information sur des corpus scientifiques (BEIR) :

| Méthode | Description | Avantage |
| :--- | :--- | :--- |
| **BM25** | Recherche lexicale par mots-clés | Baseline, rapide (< 10 ms) |
| **FAISS** | Recherche sémantique par embeddings | Comprend le sens des questions |
| **Hybride** | Fusion BM25 + FAISS | Le meilleur des deux mondes |
| **Reranking** | Cross-Encoder pour re-trier | Précision maximale (nDCG@10 0.71) |
| **RAG (Mistral)** | Génération de réponses | Synthèse automatique et contextualisée |

Le projet est **multilingue** : questions en français, recherche en anglais, résultats dans la langue du corpus.

---

## ✨ Fonctionnalités

| Fonctionnalité | Description |
| :--- | :--- |
| 🔍 **5 moteurs de recherche** | BM25, FAISS, Hybride, Rerank, RAG |
| 🌍 **Support multilingue** | Questions en français ou anglais (détection automatique) |
| 🧠 **IA générative (RAG)** | Synthèse de réponses avec Mistral-7B (température 0.1) |
| ⚡ **Cache des embeddings** | Réduction de la latence pour les questions récurrentes |
| 📜 **Historique des conversations** | Sauvegarde des 50 dernières requêtes dans la session |
| 📊 **Évaluation complète** | nDCG@10, Recall@100, latence, mémoire |
| 🚀 **API REST** | Flask avec 5 endpoints standards |
| 🖥️ **Interface web moderne** | Streamlit avec thème sombre, expanders et design pro |
| 📦 **Reproductible** | Docker, requirements.txt, Data/Model Card |
| 🔄 **Traduction automatique** | deep-translator + dictionnaire de secours |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          UTILISATEUR                                        │
│                          (Question en français)                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                    INTERFACE STREAMLIT (MODERNISÉE)                        │
│          (Barre de recherche, sélection du moteur, expanders)              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                         API FLASK                                          │
│                  Endpoints : /search_bm25, /search_faiss, ...              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MOTEURS DE RECHERCHE                               │
├─────────────┬─────────────┬─────────────┬─────────────┬────────────────────┤
│    BM25     │    FAISS    │   Hybride   │   Rerank    │        RAG         │
│  (Mots-clés)│ (Sémantique)│ (BM25+FAISS)│(Cross-Enc.) │   (Mistral-7B)     │
└─────────────┴─────────────┴─────────────┴─────────────┴────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CORPUS BEIR (Multi-domaines)                            │
│           SciFact · NFCorpus · FiQA · Arguana                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Données

### Corpus utilisés

| Dataset | Documents | Requêtes | Domaine | Licence |
| :--- | :--- | :--- | :--- | :--- |
| **SciFact** | 5 183 | 300 | Scientifique | Recherche non commerciale |
| **NFCorpus** | 3 633 | 323 | Médical | Recherche non commerciale |
| **FiQA** | 57 638 | 648 | Financier | Recherche non commerciale |
| **Arguana** | 8 674 | 1 406 | Argumentation | Recherche non commerciale |

### Index pré-générés

Les index FAISS ont été générés sur Google Colab avec le modèle `distiluse-base-multilingual-cased-v2` (dimension 512) :

| Dataset | Index | Métadonnées | Taille |
| :--- | :--- | :--- | :--- |
| SciFact | `scifact_index.index` | `scifact_metadata.pkl` | ~18 Mo |
| NFCorpus | `nfcorpus_index.index` | `nfcorpus_metadata.pkl` | ~13 Mo |
| FiQA | `fiqa_index.index` | `fiqa_metadata.pkl` | ~156 Mo |
| Arguana | `arguana_index.index` | `arguana_metadata.pkl` | ~26 Mo |

Les fichiers sont disponibles dans le dossier `models/`.

---

## 🚀 Améliorations récentes

| Amélioration | Description | Impact |
| :--- | :--- | :--- |
| **Interface Streamlit modernisée** | Thème sombre, expanders, barre de recherche centrale | UX améliorée |
| **RAG avec Mistral-7B** | Génération de réponses synthétiques via l'API Mistral | Réponse fluide et contextualisée |
| **Cache des embeddings** | Mise en cache des requêtes identiques (taille max 1000) | Réduction de la latence de 40% |
| **Historique des conversations** | Sauvegarde des 50 dernières requêtes | Rechargement rapide |
| **Support de 4 datasets** | SciFact, NFCorpus, FiQA, Arguana | Multi-domaines |
| **Traduction automatique** | deep-translator avec fallback dictionnaire | Support multilingue |
| **Expander pour résultats** | Affichage du texte intégral au clic | Lisibilité améliorée |
| **Statistiques du cache** | Affichage en temps réel du taux de hits | Transparence |

---

## 🛠️ Technologies utilisées

### Backend
| Technologie | Version | Utilisation |
| :--- | :--- | :--- |
| Python | 3.11 | Langage principal |
| Flask | 2.3.3 | API REST |
| Flask-CORS | 4.0.0 | Gestion des requêtes cross-origin |
| FAISS | 1.9.0 | Indexation vectorielle |
| Sentence-Transformers | 2.2.2 | Embeddings multilingues |
| rank-bm25 | 0.2.2 | Algorithme BM25 |
| deep-translator | 1.11.4 | Traduction automatique |
| Mistral-7B (API) | - | Génération de réponses RAG |
| Pytest | 7.4.3 | Tests automatisés |

### Frontend
| Technologie | Version | Utilisation |
| :--- | :--- | :--- |
| Streamlit | 1.29.0 | Interface web moderne |
| Base44 | - | Hébergement du site |
| CSS personnalisé | - | Thème sombre, design pro |

### Infrastructure
| Technologie | Utilisation |
| :--- | :--- |
| Docker | Conteneurisation |
| Railway | Hébergement de l'API |
| GitHub | Versionnement |
| Google Colab | Génération des index FAISS |

---

## 📦 Installation

### Prérequis
- Python 3.11
- Git
- Docker (optionnel)
- Clé API Mistral (pour le RAG)

### 1. Cloner le dépôt

```bash
git clone https://github.com/doumbia07/rag_project.git
cd rag_project
```

### 2. Créer l'environnement virtuel

```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate          # Windows
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Configurer la clé API Mistral (pour le RAG)

```bash
export MISTRAL_API_KEY="ta_clé_api_ici"   # Linux/Mac
# ou
$env:MISTRAL_API_KEY="ta_clé_api_ici"     # PowerShell
```

### 5. Placer les index FAISS

```bash
mkdir -p models
# Copier les fichiers *.index et *.pkl dans models/
```

### 6. Lancer l'API

```bash
python -m backend.app
```

### 7. Lancer l'interface Streamlit (dans un autre terminal)

```bash
streamlit run frontend/streamlit_app.py
```

---

## 🚀 Utilisation

### Interface Web (Streamlit)

1. Ouvre `http://localhost:8501`
2. Pose une question en français ou anglais
3. Sélectionne le moteur de recherche (BM25, FAISS, Hybride, Rerank, RAG)
4. Choisis le dataset (SciFact, NFCorpus, FiQA, Arguana)
5. Ajuste le nombre de résultats (Top K)
6. Consulte les résultats en expanders (clique sur le titre pour voir le texte intégral)

### Interface Web (Base44)

1. Va sur `https://the-documind.base44.app/`
2. Pose une question en français
3. Explore les résultats

### API (curl)

```bash
# BM25
curl -X POST http://localhost:5000/search_bm25 \
  -H "Content-Type: application/json" \
  -d '{"query": "Quels sont les effets de la caféine sur le sommeil ?", "top_k": 5}'

# FAISS
curl -X POST http://localhost:5000/search_faiss \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the effects of caffeine on sleep?", "top_k": 5}'

# Hybride
curl -X POST http://localhost:5000/search_hybrid \
  -H "Content-Type: application/json" \
  -d '{"query": "Quels sont les effets de la caféine ?", "top_k": 5}'

# Rerank
curl -X POST http://localhost:5000/search_rerank \
  -H "Content-Type: application/json" \
  -d '{"query": "Comment traiter l'insomnie ?", "top_k": 5}'

# RAG (avec Mistral)
curl -X POST http://localhost:5000/search_rag \
  -H "Content-Type: application/json" \
  -d '{"query": "Quels sont les effets de la caféine sur le cœur ?", "top_k": 3}'
```

---

## 🔗 API Endpoints

| Méthode | Endpoint | Description | Body JSON |
| :--- | :--- | :--- | :--- |
| GET | `/` | Informations générales | - |
| GET | `/cache/stats` | Statistiques du cache FAISS | - |
| POST | `/cache/clear` | Vider le cache FAISS | - |
| POST | `/search_bm25` | BM25 (mots-clés) | `{"query": "...", "top_k": 10, "dataset": "scifact"}` |
| POST | `/search_faiss` | FAISS (sémantique) | `{"query": "...", "top_k": 10, "dataset": "scifact"}` |
| POST | `/search_hybrid` | Hybride (BM25 + FAISS) | `{"query": "...", "top_k": 10, "dataset": "scifact"}` |
| POST | `/search_rerank` | Reranking (Cross-Encoder) | `{"query": "...", "top_k": 10, "dataset": "scifact"}` |
| POST | `/search_rag` | RAG (Mistral) | `{"query": "...", "top_k": 3, "dataset": "scifact"}` |

### Exemple de réponse (BM25)

```json
{
  "method": "BM25",
  "dataset": "scifact",
  "query_original": "Quels sont les effets de la caféine sur le sommeil ?",
  "query_used": "What are the effects of caffeine on sleep?",
  "total_results": 10,
  "time_ms": 5.23,
  "results": [
    {
      "doc_id": "MED-880",
      "title": "Effects of acute administration of caffeine on vascular function.",
      "text": "Caffeine is the most widely used pharmacologic substance...",
      "score": 18.1663
    }
  ]
}
```

---

## 📊 Évaluation

### Métriques
| Métrique | Description |
| :--- | :--- |
| **nDCG@10** | Normalized Discounted Cumulative Gain |
| **Recall@100** | Proportion de documents pertinents retrouvés |
| **MRR** | Mean Reciprocal Rank |
| **Latence** | Temps de réponse moyen (ms) |
| **Mémoire** | Utilisation mémoire (MB) |
| **Cache hit rate** | Taux de réussite du cache des embeddings |

### Résultats (SciFact, 300 requêtes)

| Méthode | nDCG@10 | Recall@100 | Latence (ms) | Mémoire (MB) |
| :--- | :--- | :--- | :--- | :--- |
| BM25 | 0.42 | 0.65 | 5 | 150 |
| FAISS | 0.56 | 0.78 | 15 | 500 |
| Hybride | 0.63 | 0.85 | 20 | 650 |
| Rerank | 0.71 | 0.89 | 120 | 1200 |
| RAG (Mistral) | - | - | 800-2000 | 2000 |

### Statistiques du cache
- **Cache activé** : ✅
- **Taille max** : 1000 embeddings
- **Taux de hits typique** : 60-80% (questions récurrentes)

---

## 🌐 Déploiement

### Railway (API Flask)

1. Créer un compte sur [Railway.app](https://railway.app/)
2. Connecter le dépôt GitHub
3. Déployer automatiquement
4. Configurer la variable d'environnement `MISTRAL_API_KEY`

### Base44 (Site web)

1. Connecter le projet sur [Base44.com](https://base44.com/)
2. Configurer la variable d'environnement `VITE_API_URL` avec l'URL de l'API
3. Déployer

### Docker

```bash
docker build -t rag-engine .
docker run -p 5000:5000 -e MISTRAL_API_KEY="ta_clé" rag-engine
```

---

## 📚 Documentation

| Document | Description |
| :--- | :--- |
| [DATA_CARD.md](./DATA_CARD.md) | Fiche des données (source, licence, biais) |
| [MODEL_CARD.md](./MODEL_CARD.md) | Fiche des modèles (performances, limitations) |
| [Rapport final](./docs/RAPPORT_FINAL.md) | Analyse complète du projet |

---

## 📦 Livrables

- ✅ **Dépôt GitHub** : README, code, licence
- ✅ **Data Card** : Source, licence, transformations, biais
- ✅ **Model Card** : Description des modèles, performances, limites
- ✅ **API Flask** : 5 endpoints (BM25, FAISS, Hybride, Rerank, RAG)
- ✅ **Interface web Streamlit** : Design moderne, thème sombre, expanders
- ✅ **Interface web Base44** : Hébergée en ligne
- ✅ **Rapport d'évaluation** : Tableaux comparatifs, analyse d'erreurs
- ✅ **Tests automatisés** : Pytest
- ✅ **Déploiement** : Railway + Base44
- ✅ **Cache des embeddings** : Réduction de la latence
- ✅ **Historique des conversations** : Sauvegarde et rechargement
- ✅ **RAG avec Mistral-7B** : Génération de réponses synthétiques

---

## 🧑‍💻 Auteurs

**Doumbia Abou-Bakar Sidik** - Projet de fin de formation  
Laboratoire de recherche sémantique multi-domaines

---

## 📄 Licence

Ce projet est sous licence **MIT**. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

**Note importante** : Les données BEIR sont sous licence **recherche non commerciale**. Consultez les licences respectives avant toute utilisation commerciale.

---

## 🙏 Remerciements

- [BEIR](https://github.com/beir-cellar/beir) pour les datasets
- [Sentence-Transformers](https://www.sbert.net/) pour les embeddings
- [FAISS](https://github.com/facebookresearch/faiss) pour l'indexation vectorielle
- [Streamlit](https://streamlit.io/) pour l'interface web
- [Base44](https://base44.com/) pour l'hébergement
- [Mistral AI](https://mistral.ai/) pour l'API de génération

---

## 🔗 Liens utiles

- [Dépôt GitHub](https://github.com/doumbia07/rag_project)
- [Site Base44](https://the-documind.base44.app/)
- [API déployée](https://rag-project.up.railway.app/)
- [Data Card](./DATA_CARD.md)
- [Model Card](./MODEL_CARD.md)

---

*Projet réalisé dans le cadre d'une formation en Intelligence Artificielle - Septembre 2026*