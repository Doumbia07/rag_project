

rag_project/                          # Racine de ton projet
│
├── backend/                          # ⚙️ Cœur du projet (API + moteurs)
│   ├── __init__.py                   # Fichier vide (indique que c'est un package Python)
│   ├── app.py                        # Serveur Flask (les endpoints /search_*)
│   ├── data_loader.py                # Téléchargement et chargement des données BEIR
│   ├── bm25.py                       # Moteur de recherche BM25 (baseline)
│   ├── faiss_search.py               # Moteur avec embeddings + FAISS
│   ├── hybrid.py                     # Fusion BM25 + FAISS
│   ├── rerank.py                     # Cross-Encoder pour reranking
│   ├── rag.py                        # Pipeline RAG (génération de réponse)
│   └── evaluator.py                  # Calcul des métriques (nDCG, Recall, latence)
│
├── frontend/                         # 🖥️ Interface utilisateur (Streamlit)
│   └── streamlit_app.py              # Application Streamlit (appelle l'API Flask)
│
├── data/                             # 📊 Datasets BEIR (créé automatiquement)
│   └── (scifact/, nfcorpus/, etc.)   # Dossiers téléchargés par data_loader.py
│
├── notebooks/                        # 📓 Exploration et tests (optionnel)
│   └── exploration.ipynb             # Notebook Jupyter pour tester des idées
│
├── tests/                            # ✅ Tests automatisés (Pytest)
│   ├── test_api.py                   # Teste les endpoints de l'API
│   └── test_bm25.py                  # Teste le moteur BM25
│
├── .gitignore                        # Fichiers à ne pas versionner sur GitHub
├── requirements.txt                  # Dépendances Python (Flask, FAISS, etc.)
├── Dockerfile                        # Déploiement sur Render/Railway
├── render.yaml                       # Configuration pour Render (optionnel)
├── README.md                         # Présentation du projet, installation, usage
├── DATA_CARD.md                      # Fiche des données (source, licence, biais)
└── MODEL_CARD.md                     # Fiche des modèles (BM25, embeddings, etc.)

# 🔍 RAG Multi-Domain - Moteur de recherche intelligent

**Laboratoire de recherche sémantique multi-domaines** pour l'évaluation et la comparaison de moteurs de recherche.

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3.3-green.svg)](https://flask.palletsprojects.com/)
[![FAISS](https://img.shields.io/badge/FAISS-1.9.0-orange.svg)](https://github.com/facebookresearch/faiss)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29.0-red.svg)](https://streamlit.io/)

---

## 📋 Table des matières

- [Présentation du projet](#-présentation-du-projet)
- [Fonctionnalités](#-fonctionnalités)
- [Architecture](#-architecture)
- [Données](#-données)
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
| **BM25** | Recherche lexicale par mots-clés | Baseline, rapide |
| **FAISS** | Recherche sémantique par embeddings | Comprend le sens |
| **Hybride** | Fusion BM25 + FAISS | Le meilleur des deux |
| **Reranking** | Cross-Encoder pour re-trier | Précision maximale |
| **RAG** | Génération de réponses | Synthèse automatique |

Le projet est **multilingue** : questions en français, recherche en anglais, résultats dans la langue du corpus.

---

## ✨ Fonctionnalités

| Fonctionnalité | Description |
| :--- | :--- |
| 🔍 **Recherche multi-moteurs** | BM25, FAISS, Hybride, Rerank, RAG |
| 🌍 **Support multilingue** | Questions en français ou anglais |
| 📊 **Évaluation complète** | nDCG@10, Recall@100, latence, mémoire |
| 🚀 **API REST** | Flask avec endpoints standards |
| 🖥️ **Interface web** | Streamlit et Base44 |
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
│                    INTERFACE STREAMLIT / BASE44                            │
│                    (Saisie, sélection du moteur, affichage)                │
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
│  (Mots-clés)│ (Sémantique)│ (BM25+FAISS)│(Cross-Enc.) │   (Génération)     │
└─────────────┴─────────────┴─────────────┴─────────────┴────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CORPUS BEIR                                        │
│              SciFact (5 183 documents) / NFCorpus                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Données

### Corpus utilisés

| Dataset | Documents | Requêtes | Domaine | Licence |
| :--- | :--- | :--- | :--- | :--- |
| **SciFact** | 5 183 | 300 | Scientifique | Recherche non commerciale |
| **NFCorpus** | 3 633 | 323 | Médical | Recherche non commerciale |

### Index pré-générés

L'index FAISS pour **SciFact (5 183 documents)** a été généré sur Google Colab avec le modèle `distiluse-base-multilingual-cased-v2` (dimension 512). Les fichiers sont disponibles dans le dossier `models/` :

- `faiss_index.index` : Index FAISS (~80 Mo)
- `metadata.pkl` : Métadonnées (doc_ids, titres, textes)

---

## 🛠️ Technologies utilisées

### Backend
| Technologie | Version | Utilisation |
| :--- | :--- | :--- |
| Python | 3.11 | Langage principal |
| Flask | 2.3.3 | API REST |
| FAISS | 1.9.0 | Indexation vectorielle |
| Sentence-Transformers | 2.2.2 | Embeddings multilingues |
| rank-bm25 | 0.2.2 | Algorithme BM25 |
| deep-translator | 1.11.4 | Traduction automatique |

### Frontend
| Technologie | Version | Utilisation |
| :--- | :--- | :--- |
| Streamlit | 1.29.0 | Interface web |
| Base44 | - | Hébergement du site |

### Infrastructure
| Technologie | Utilisation |
| :--- | :--- |
| Docker | Conteneurisation |
| Railway | Hébergement de l'API |
| GitHub | Versionnement |

---

## 📦 Installation

### Prérequis
- Python 3.11
- Git
- Docker (optionnel)

### 1. Cloner le dépôt

```bash
git clone https://github.com/ton-username/rag_project.git
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

### 4. Placer les index FAISS (optionnel)

Si tu utilises l'index pré-généré pour SciFact :

```bash
mkdir -p models
# Place faiss_index.index et metadata.pkl dans models/
```

### 5. Lancer l'API

```bash
python -m backend.app
```

### 6. Lancer l'interface Streamlit (dans un autre terminal)

```bash
streamlit run frontend/streamlit_app.py
```

---

## 🚀 Utilisation

### Interface Web (Streamlit)

1. Ouvre `http://localhost:8501`
2. Pose une question en français ou anglais
3. Sélectionne le moteur de recherche
4. Ajuste le nombre de résultats
5. Consulte les résultats avec scores et sources

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

# RAG
curl -X POST http://localhost:5000/search_rag \
  -H "Content-Type: application/json" \
  -d '{"query": "Quels sont les effets de la caféine sur le cœur ?", "top_k": 3}'
```

---

## 🔗 API Endpoints

| Méthode | Endpoint | Description | Body JSON |
| :--- | :--- | :--- | :--- |
| GET | `/` | Informations générales | - |
| POST | `/search_bm25` | BM25 (mots-clés) | `{"query": "...", "top_k": 10}` |
| POST | `/search_faiss` | FAISS (sémantique) | `{"query": "...", "top_k": 10}` |
| POST | `/search_hybrid` | Hybride (BM25 + FAISS) | `{"query": "...", "top_k": 10}` |
| POST | `/search_rerank` | Reranking (Cross-Encoder) | `{"query": "...", "top_k": 10}` |
| POST | `/search_rag` | RAG (génération) | `{"query": "...", "top_k": 3}` |

### Exemple de réponse

```json
{
  "method": "BM25",
  "query_original": "Quels sont les effets de la caféine sur le sommeil ?",
  "query_translated": "What are the effects of caffeine on sleep?",
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

### Résultats (SciFact, 300 requêtes)

| Méthode | nDCG@10 | Recall@100 | Latence (ms) | Mémoire (MB) |
| :--- | :--- | :--- | :--- | :--- |
| BM25 | 0.42 | 0.65 | 5 | 150 |
| FAISS | 0.56 | 0.78 | 15 | 500 |
| Hybride | 0.63 | 0.85 | 20 | 650 |
| Rerank | 0.71 | 0.89 | 120 | 1200 |
| RAG | - | - | 800 | 2000 |

---

## 🌐 Déploiement

### Railway (API Flask)

1. Créer un compte sur [Railway.app](https://railway.app/)
2. Connecter le dépôt GitHub
3. Déployer automatiquement

### Base44 (Site web)

1. Connecter le projet sur [Base44.com](https://base44.com/)
2. Configurer la variable d'environnement `VITE_API_URL`
3. Déployer

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
- ✅ **Interface web** : Streamlit + Base44
- ✅ **Rapport d'évaluation** : Tableaux comparatifs, analyse d'erreurs
- ✅ **Tests automatisés** : Pytest
- ✅ **Déploiement** : Railway + Base44

---

## 🧑‍💻 Auteurs

**Karim** - Projet de fin de formation  
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

---

## 🔗 Liens utiles

- [Dépôt GitHub](https://github.com/doumbia07/rag_project)
- [Site Base44](https://the-documind.base44.app/)
- [API déployée](https://rag-project.up.railway.app/)
- [Data Card](./DATA_CARD.md)
- [Model Card](./MODEL_CARD.md)

---

*Projet réalisé dans le cadre d'une formation en Intelligence Artificielle.*