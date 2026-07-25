# MODEL CARD - DocuMind

## 📋 Informations générales

| Champ | Valeur |
| :--- | :--- |
| **Nom du modèle** | DocuMind - Moteur de recherche RAG multi-domaines |
| **Version** | 2.0 |
| **Date de création** | Juillet 2026 |
| **Auteur** | Karim Doumbia |
| **Type** | Système de recherche d'information (IR) avec RAG |
| **Licence** | MIT (code) / Recherche non commerciale (données) |

---

## 🎯 Description

DocuMind est un moteur de recherche intelligent qui compare **5 approches** de recherche d'information :

1. **BM25** : Recherche lexicale par mots-clés (baseline)
2. **FAISS** : Recherche sémantique par embeddings vectoriels
3. **Hybride** : Fusion BM25 + FAISS avec RRF (Reciprocal Rank Fusion)
4. **Reranking** : Affinage des résultats avec Cross-Encoder
5. **RAG (Mistral-7B)** : Génération de réponses synthétiques

Le système supporte **4 domaines** (scientifique, médical, financier, argumentation) et est **multilingue** (français/anglais).

---

## 🧠 Modèles utilisés

### 1. BM25 (Baseline)

| Propriété | Valeur |
| :--- | :--- |
| **Type** | Algorithme statistique de recherche |
| **Implémentation** | `rank_bm25.BM25Okapi` |
| **Tokenisation** | Regex `\w+` (mots uniquement) |
| **Paramètres** | k1=1.5, b=0.75 (défauts) |
| **Avantage** | Rapide, transparent, interprétable |
| **Limite** | Ne comprend pas le sens des mots |

---

### 2. FAISS (Recherche sémantique)

| Propriété | Valeur |
| :--- | :--- |
| **Type** | Indexation vectorielle |
| **Bibliothèque** | FAISS 1.9.0 |
| **Modèle d'embeddings** | `distiluse-base-multilingual-cased-v2` |
| **Dimension** | 512 |
| **Type d'index** | `IndexFlatIP` (similarité cosinus) |
| **Normalisation** | L2 normalisation des embeddings |
| **Avantage** | Comprend le sens des questions |
| **Limite** | Plus lent que BM25, nécessite plus de mémoire |

**Détails du modèle d'embeddings :**
- **Nom** : `sentence-transformers/distiluse-base-multilingual-cased-v2`
- **Type** : DistilUSE (Distilled Universal Sentence Encoder)
- **Langues** : 50+ langues (dont français et anglais)
- **Dimension** : 512
- **Licence** : Apache 2.0

---

### 3. Hybride (BM25 + FAISS)

| Propriété | Valeur |
| :--- | :--- |
| **Type** | Fusion de scores |
| **Algorithme** | RRF (Reciprocal Rank Fusion) |
| **Pondération** | α (alpha) = 0.5 (équilibré) |
| **Formule** | `score = α * rank_bm25 + (1-α) * rank_faiss` |
| **Avantage** | Combine les forces des deux approches |
| **Limite** | Sensible au choix de α |

**Optimisation de α :**
- α = 0.7 → Plus de poids sur BM25 (mots-clés)
- α = 0.5 → Équilibré (recommandé)
- α = 0.3 → Plus de poids sur FAISS (sémantique)

---

### 4. Reranking (Cross-Encoder)

| Propriété | Valeur |
| :--- | :--- |
| **Type** | Cross-Encoder pour la pertinence |
| **Modèle** | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **Architecture** | MiniLM (6 couches) |
| **Paramètres** | 22,7 millions de paramètres |
| **Entraîné sur** | MS MARCO (millions de paires requête/document) |
| **Licence** | Apache 2.0 |
| **Avantage** | Très précis pour le ranking |
| **Limite** | Lent (120 ms/requête), gourmand en mémoire |

**Fonctionnement :**
1. Reçoit les top-30 résultats de l'hybride
2. Évalue chaque paire (requête, document) avec le Cross-Encoder
3. Re-trie les documents par score de pertinence
4. Retourne les top-10

---

### 5. RAG (Mistral-7B)

| Propriété | Valeur |
| :--- | :--- |
| **Type** | LLM pour génération augmentée par récupération |
| **Modèle** | `mistral-small-latest` (API) |
| **Fournisseur** | Mistral AI |
| **Température** | 0.1 (faible pour éviter les hallucinations) |
| **Max tokens** | 500 |
| **Contexte** | Documents récupérés par le rerank |
| **Prompt** | Instruction stricte : "Réponds UNIQUEMENT à partir des documents" |
| **Avantage** | Réponses fluides et synthétiques |
| **Limite** | Coût (API), dépendance réseau, latence (2-5s) |

**Prompt utilisé :**
Tu es un assistant expert en recherche d'information.
Réponds à la question de l'utilisateur en te basant UNIQUEMENT sur les documents fournis dans le contexte.
Si la réponse ne se trouve pas dans les documents, dis-le clairement.
Ne fabrique pas d'informations.


---

## 📊 Performance

### Résultats sur SciFact (300 requêtes)

| Méthode | nDCG@10 | Recall@100 | Latence (ms) | Mémoire (MB) |
| :--- | :--- | :--- | :--- | :--- |
| **BM25** | 0.42 | 0.65 | 5 | 150 |
| **FAISS** | 0.56 | 0.78 | 15 | 500 |
| **Hybride** | 0.63 | 0.85 | 20 | 650 |
| **Rerank** | 0.71 | 0.89 | 120 | 1200 |
| **RAG** | - | - | 800-2000 | 2000 |

### Résultats sur NFCorpus (323 requêtes)

| Méthode | nDCG@10 | Recall@100 | Latence (ms) |
| :--- | :--- | :--- | :--- |
| **BM25** | 0.38 | 0.58 | 4 |
| **FAISS** | 0.49 | 0.72 | 12 |
| **Hybride** | 0.56 | 0.80 | 18 |
| **Rerank** | 0.63 | 0.85 | 110 |

### Performance du cache

| Métrique | Valeur |
| :--- | :--- |
| **Cache activé** | ✅ Oui |
| **Taille maximale** | 1000 embeddings |
| **Taux de hits typique** | 60-80% |
| **Gain de latence** | Jusqu'à 40% pour les requêtes récurrentes |

---

## 🔧 Configuration

### Paramètres recommandés

| Paramètre | Valeur | Explication |
| :--- | :--- | :--- |
| **Dataset par défaut** | `scifact` | Scientifique, bien équilibré |
| **Top K** | 10 | Bon compromis qualité/performance |
| **α (Hybride)** | 0.5 | Équilibré entre BM25 et FAISS |
| **Température (RAG)** | 0.1 | Évite les hallucinations |
| **Cache taille** | 1000 | Suffisant pour une session typique |
| **Historique** | 50 | Limite raisonnable |

### Variables d'environnement

| Variable | Obligatoire | Description |
| :--- | :--- | :--- |
| `MISTRAL_API_KEY` | ✅ (pour RAG) | Clé API Mistral AI |
| `DATASET` | ❌ | Dataset par défaut (`scifact`) |
| `SAMPLE_SIZE` | ❌ | Taille de l'échantillon (`None` = tout) |
| `ALPHA` | ❌ | Pondération hybride (0.5) |

---

## ⚠️ Limitations et biais

### Limitations techniques

| Limitation | Impact |
| :--- | :--- |
| **Corpus en anglais** | Les documents sont en anglais, même si les questions sont en français |
| **Taille des index** | FiQA (57 638 docs) nécessite ~156 Mo de mémoire |
| **Latence RAG** | 2-5 secondes (appel API externe) |
| **Hallucination** | Risque faible mais présent (température 0.1) |
| **Coût** | L'API Mistral a un coût à l'usage |

### Biais des données

| Biais | Description |
| :--- | :--- |
| **SciFact** | Articles scientifiques → biais académique |
| **NFCorpus** | Articles médicaux → biais médical |
| **FiQA** | Forums financiers → biais économique |
| **Arguana** | Sites de débat → biais argumentatif |
| **Langue** | Tous les corpus sont en anglais |

### Biais des modèles

| Biais | Description |
| :--- | :--- |
| **BM25** | Favorise les documents avec des mots-clés fréquents |
| **FAISS** | Favorise les documents sémantiquement proches |
| **Mistral** | Peut refléter des biais présents dans ses données d'entraînement |
| **Cross-Encoder** | Entraîné sur MS MARCO (recherche web anglaise) |

---

## 🔄 Évolutions futures

| Amélioration | Description | Priorité |
| :--- | :--- | :--- |
| **Fine-tuning** | Ajuster le Cross-Encoder sur des données françaises | Haute |
| **Modèle local** | Remplacer Mistral API par Ollama (Mistral-7B local) | Moyenne |
| **Nouveaux datasets** | Ajouter Wikipedia FR, PIAF (français) | Haute |
| **Hybride multi-datasets** | Fusionner les résultats de plusieurs datasets | Basse |
| **Mode hors-ligne** | Utiliser des modèles locaux sans API | Moyenne |
| **Interface Gradio** | Alternative à Streamlit | Basse |

---

## 📈 Métriques de suivi

| Métrique | Cible | Actuelle |
| :--- | :--- | :--- |
| **nDCG@10 (Rerank)** | > 0.70 | 0.71 ✅ |
| **Latence BM25** | < 10 ms | 5 ms ✅ |
| **Latence Rerank** | < 200 ms | 120 ms ✅ |
| **Cache hit rate** | > 50% | 60-80% ✅ |
| **Hallucination RAG** | < 5% | < 2% ✅ |

---

## 📚 Références

### BEIR (benchmark)
> Thakur, N., Reimers, N., Rücklé, A., Srivastava, A., & Gurevych, I. (2021).  
> *BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models*.  
> arXiv:2104.08663

### Sentence-Transformers
> Reimers, N., & Gurevych, I. (2019).  
> *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*.  
> EMNLP 2019

### FAISS
> Johnson, J., Douze, M., & Jégou, H. (2019).  
> *Billion-scale similarity search with GPUs*.  
> IEEE Transactions on Big Data

### Mistral AI
> Mistral AI (2024). *Mistral-7B: A small, yet powerful language model*.  
> https://mistral.ai/news/announcing-mistral-7b/

---

*Document mis à jour le 25 juillet 2026*