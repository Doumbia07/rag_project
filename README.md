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