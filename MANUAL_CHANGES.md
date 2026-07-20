MANUEL COMPLET DES MODIFICATIONS
=================================

Résumé
------
Ce document décrit exhaustivement toutes les modifications que j'ai effectuées dans ce dépôt pour rendre l'API Flask RAG fonctionnelle localement, corriger le parsing JSON, gérer des archives BEIR corrompues, résoudre des incompatibilités mineures de dépendances et améliorer la robustesse du chargement des métadonnées FAISS.

Contexte initial
----------------
- Projet: RAG search backend (Flask) + frontend Streamlit.
- Problèmes signalés par l'utilisateur:
  - Requêtes POST `curl` retournant JSON invalide côté API.
  - Téléchargement BEIR produisait un `scifact.zip` corrompu (BadZipFile).
  - ImportError lié à `huggingface_hub.is_offline_mode` lors de l'import de `sentence-transformers`/`transformers`.
  - Endpoints `/search_*` ne fonctionnaient pas tous à cause des problèmes ci-dessus.

Actions effectuées (liste complète)
-----------------------------------
1) Robustification du parsing JSON et des endpoints Flask
   - Fichier modifié: [backend/app.py](backend/app.py)
   - Changements clés:
     - Ajout d'une fonction `parse_request_json()` qui essaye `request.get_json(silent=True)` puis fait un fallback en lisant `request.get_data(as_text=True)` et `json.loads` pour mieux gérer différentes formes de corps HTTP (notamment problèmes de quoting sous Windows / curl).
     - Les routes `/search_bm25`, `/search_faiss`, `/search_hybrid`, `/search_rerank`, `/search_rag` utilisent désormais `parse_request_json()` et `validate_request()` de façon uniforme.
     - Ajout d'un shim de compatibilité pour `huggingface_hub.is_offline_mode` avant d'importer `sentence-transformers`/`transformers` dépendances: si la fonction manque, on injecte un fallback retournant `False`.
     - Renforcement des gestionnaires d'erreurs (logs + messages JSON clairs).
   - But: éviter les erreurs de parsing JSON et l'ImportError lors du démarrage.

2) Gestion des datasets BEIR corrompus
   - Fichier modifié: [backend/data_loader.py](backend/data_loader.py)
   - Changements clés:
     - Ajout de détection de `*.zip` corrompus via `zipfile.is_zipfile()`; suppression automatique du zip corrompu avant retéléchargement.
     - Vérification que le dossier extrait contient `corpus.jsonl`; si absent, suppression du dossier incomplet et redémarrage du téléchargement.
     - Après le téléchargement, levée d'une `FileNotFoundError` si le dossier attendu n'existe pas (fail fast).
   - But: rendre robuste le flux `util.download_and_unzip()` face à des archives partielles/corrompues et éviter des échecs en cascade.

3) Acceptation de variantes de format metadata FAISS
   - Fichier modifié: [backend/faiss_search.py](backend/faiss_search.py)
   - Changements clés:
     - `_load_metadata()` désormais accepte plusieurs structures courantes pour `metadata.pkl`:
       - `{ 'doc_ids', 'doc_titles', 'doc_texts' }` (clé présente dans `models/metadata.pkl` de l'utilisateur)
       - `{ 'doc_ids', 'titles', 'texts' }`
       - mapping `doc_id -> { 'title', 'text' }`
       - `{ 'corpus': { ... } }`
     - Si aucune des formes reconnues n'est détectée, lève une erreur explicite.
   - But: éviter l'échec si le `metadata.pkl` vient de différentes sources/outils.

4) Fix d'import-time pour `huggingface_hub`
   - Fichier modifié: [backend/app.py](backend/app.py) (shim mentionné ci-dessus)
   - Détail: certaines versions de `transformers`/`sentence-transformers` attendent `is_offline_mode` exporté par `huggingface_hub`. Le shim injecte une fonction minimale si manquante.
   - But: permettre l'import de `sentence_transformers` sans modifier l'environnement immédiatement.

5) Pin de dépendance
   - Fichier modifié: `requirements.txt`
   - Ajout: `huggingface_hub==0.16.4`
   - But: stabiliser l'environnement et éviter régressions futures liées à changements breaking dans `huggingface_hub`.

6) Commits
   - Commits réalisés (extraits):
     - "Fix JSON parsing; handle corrupted BEIR zip; accept alternate FAISS metadata keys; add HF hub shim"
       - Fichiers modifiés: `backend/app.py`, `backend/data_loader.py`, `backend/faiss_search.py`
     - "Pin huggingface_hub to 0.16.4 in requirements.txt"
       - Fichier modifié: `requirements.txt`

7) Tests et vérifications
   - Utilisation du `Flask` test client pour envoyer POST locaux aux endpoints `/search_*`.
   - Résultats observés localement (extraits):
     - `/search_bm25`: renvoie JSON avec `results` — BM25 index construit correctement après chargement du dataset.
     - `/search_faiss`: charge l'index FAISS depuis `models/faiss_index.index` et `models/metadata.pkl` (format détecté `doc_ids/doc_titles/doc_texts`), charge le modèle d'embeddings, renvoie résultats.
     - `/search_hybrid` et `/search_rerank`: initialisent BM25+FAISS et Cross-Encoder et renvoient des résultats.
     - `/search_rag`: utilise le reranking et génère une réponse synthétique plus courte.
   - Note: les fichiers `tests/test_bm25.py` et `tests/test_api.py` étaient vides — pytest n'a lancé aucun test automatisé. J'ai effectué des tests manuels via client Flask.

Emplacement des fichiers modifiés
--------------------------------
- [backend/app.py](backend/app.py)
- [backend/data_loader.py](backend/data_loader.py)
- [backend/faiss_search.py](backend/faiss_search.py)
- [requirements.txt](requirements.txt)
- Nouveau fichier de manuel créé: `MANUAL_CHANGES.md` (vous lisez ce fichier)

Détails techniques (extraits et explications)
--------------------------------------------
1) Parsing JSON — `backend/app.py`
   - Nouvelle fonction:
     ```py
     def parse_request_json():
         data = request.get_json(silent=True)
         if data is None:
             try:
                 raw_body = request.get_data(as_text=True)
                 if raw_body:
                     data = json.loads(raw_body)
             except Exception:
                 data = None
         return data
     ```
   - Usage: chaque endpoint appelle `data = parse_request_json()` puis `validate_request(data)`.
   - Pourquoi: `curl` sur Windows ou PowerShell peut produire des payloads qui ne sont pas correctement dé-serialisés par `get_json()` — fallback lisant `request.get_data()` et `json.loads()` résout cela.

2) BEIR dataset handling — `backend/data_loader.py`
   - Check zip corrompu:
     ```py
     if os.path.exists(zip_path) and not zipfile.is_zipfile(zip_path):
         os.remove(zip_path)
     ```
   - Check dossier extrait complet (présence de `corpus.jsonl`) et suppression si incomplet.
   - Appel `util.download_and_unzip(url, DATA_DIR)` pour retélécharger proprement.

3) FAISS metadata loader — `backend/faiss_search.py`
   - Acceptation des clés `doc_titles/doc_texts` et autres formats.
   - Raison: l'archive `models/metadata.pkl` fournie dans l'environnement avait les clés `doc_titles`/`doc_texts` au lieu de `titles`/`texts`, provoquant une ValueError.

4) Shim `huggingface_hub.is_offline_mode` — `backend/app.py`
   - Code injecté (simplifié):
     ```py
     try:
         import huggingface_hub
         if not hasattr(huggingface_hub, 'is_offline_mode'):
             def _hf_is_offline_mode():
                 return False
             setattr(huggingface_hub, 'is_offline_mode', _hf_is_offline_mode)
     except Exception:
         pass
     ```
   - Ceci évite l'ImportError intercepté précédemment quand `sentence_transformers` importait `transformers`.

Comment reproduire localement
-----------------------------
1) (Optionnel) Créer un venv propre et installer dépendances:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

2) Lancer l'API Flask en local:

```bash
python -m backend.app
```

3) Exemple de requête curl (PowerShell / Windows):

```powershell
curl.exe -H "Content-Type: application/json" -d '{ "query": "quels sont les effets du fer sur l organisme ?", "top_k": 3 }' http://localhost:5000/search_bm25
```

4) Lancer la UI Streamlit (optionnel):

```bash
streamlit run frontend/streamlit_app.py
```

Tests effectués
---------------
- Tests manuels via `Flask.test_client()` (script ad-hoc exécuté dans l'environnement). Les endpoints retournent JSON valides.
- `pytest` n'a rien exécuté car les fichiers dans `tests/` étaient vides.

Recommandations et prochaines étapes
-----------------------------------
1) Ajouter de vrais tests unitaires dans `tests/` pour couvrir:
   - `backend/data_loader.load_dataset()` (mocker download si besoin)
   - `backend/bm25.BM25Search.search()` (utiliser `SAMPLE_SIZE` pour dataset réduit)
   - Endpoints API via `Flask` test client

2) Vérifier l'environnement de production et décider si le shim HF doit être conservé ou remplacé par une contrainte stricte de version; le pin dans `requirements.txt` (fait) est recommandé en environnement dev.

3) Ajouter des instructions d'installation plus détaillées dans `README.md` si souhaité (j'ajoute si vous validez).

4) Commit & push: j'ai commité localement les changements. Si vous souhaitez que je pousse vers un remote (GitHub), donnez l'autorisation et vérifiez que le remote est configuré.

Historique des commandes et actions utiles
-----------------------------------------
- Vérifier la présence des zips corrompus:

```bash
python - <<'PY'
import zipfile, os
print(zipfile.is_zipfile('data/scifact.zip'))
PY
```

- Forcer la suppression et retéléchargement (script ad-hoc fait pendant le debug): j'ai supprimé `data/scifact.zip` localement et appelé `_ensure_dataset('scifact')` pour retélécharger et extraire proprement.

Fichiers exacts modifiés
------------------------
- [backend/app.py](backend/app.py) — parsing JSON, HF shim, endpoint error handling
- [backend/data_loader.py](backend/data_loader.py) — zip integrity checks, clean re-download
- [backend/faiss_search.py](backend/faiss_search.py) — flexible metadata loader
- [requirements.txt](requirements.txt) — added `huggingface_hub==0.16.4`
- [MANUAL_CHANGES.md](MANUAL_CHANGES.md) — ce document (créé)

Remarques finales
-----------------
- Tout est fonctionnel localement d'après mes tests manuels. La robustesse a été améliorée mais il reste conseillé d'ajouter des tests unitaires pour prévenir les régressions.
- Si vous voulez, je peux maintenant:
  - ajouter des tests unitaires basiques et les exécuter, ou
  - pousser les commits sur le remote GitHub, ou
  - enrichir le `README.md` avec étapes d'installation et d'utilisation.


---
Faites-moi savoir quelle action vous voulez que je fasse ensuite (ajouter des tests, push vers remote, améliorer README, etc.).
