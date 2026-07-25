import streamlit as st
import requests
from datetime import datetime
import time

# ============================================================
# CONFIGURATION PAGE
# ============================================================
st.set_page_config(
    page_title="DocuMind - Moteur de recherche RAG",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CSS PERSONNALISÉ (THÈME SOMBRE HARMONIEUX)
# ============================================================
st.markdown("""
<style>
    /* ===== POLICE ===== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, sans-serif;
    }


    /* ===== FOND GLOBAL ===== */
    .stApp {
        background-color: #0f172a !important;
    }
    .stApp > div:first-child {
        background-color: #0f172a !important;
    }
    .main > div {
        background-color: #0f172a !important;
    }

    /* ===== SIDEBAR ===== */
    section[data-testid="stSidebar"] {
        background-color: #1e293b !important;
        border-right: 1px solid #334155 !important;
        padding: 1.5rem 0.5rem !important;
    }
    section[data-testid="stSidebar"] .stMarkdown h3 {
        font-size: 0.75rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #94a3b8 !important;
        margin: 1.5rem 0 0.5rem 0 !important;
        padding-bottom: 0.3rem !important;
        border-bottom: 1px solid #334155 !important;
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stTextInput label {
        font-size: 0.7rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #94a3b8 !important;
        font-weight: 600 !important;
    }
    section[data-testid="stSidebar"] .stSelectbox > div,
    section[data-testid="stSidebar"] .stSlider > div,
    section[data-testid="stSidebar"] .stTextInput > div {
        margin-top: -0.3rem !important;
    }
    section[data-testid="stSidebar"] .stSelectbox select,
    section[data-testid="stSidebar"] .stTextInput input {
        background-color: #0f172a !important;
        color: #f1f5f9 !important;
        border-color: #334155 !important;
        border-radius: 8px !important;
    }
    section[data-testid="stSidebar"] .stSelectbox select:focus,
    section[data-testid="stSidebar"] .stTextInput input:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2) !important;
    }
    section[data-testid="stSidebar"] .stSlider > div {
        color: #f1f5f9 !important;
    }
    section[data-testid="stSidebar"] .stSlider .stSliderValue {
        color: #f1f5f9 !important;
    }
    section[data-testid="stSidebar"] .stButton button {
        border-radius: 8px !important;
        font-weight: 500 !important;
        font-size: 0.8rem !important;
        padding: 0.4rem 0.8rem !important;
        transition: all 0.15s ease !important;
        background: #0f172a !important;
        color: #f1f5f9 !important;
        border: 1px solid #334155 !important;
    }
    section[data-testid="stSidebar"] .stButton button:hover {
        background: #2563eb !important;
        border-color: #2563eb !important;
        color: white !important;
    }
    section[data-testid="stSidebar"] .stMetric {
        background-color: #0f172a !important;
        border-radius: 8px !important;
        padding: 0.4rem 0.6rem !important;
        border: 1px solid #334155 !important;
    }
    section[data-testid="stSidebar"] .stMetric label {
        color: #94a3b8 !important;
    }
    section[data-testid="stSidebar"] .stMetric .stMetricValue {
        color: #f1f5f9 !important;
    }
    section[data-testid="stSidebar"] .stInfo {
        background-color: #0f172a !important;
        border-color: #334155 !important;
        color: #94a3b8 !important;
    }

    /* ===== HEADER ===== */
    .main-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.5rem 0 1rem 0;
        border-bottom: 1px solid #334155;
        margin-bottom: 1.5rem;
    }
    .main-header .logo {
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .main-header .logo h1 {
        font-size: 1.8rem;
        font-weight: 800;
        margin: 0;
        color: #f1f5f9;
        letter-spacing: -0.5px;
    }
    .main-header .logo span {
        font-size: 1.8rem;
    }
    .main-header .badge {
        background: #3b82f6;
        color: white;
        font-size: 0.55rem;
        font-weight: 600;
        padding: 0.15rem 0.6rem;
        border-radius: 12px;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .main-header .status {
        display: flex;
        align-items: center;
        gap: 0.4rem;
        font-size: 0.75rem;
        color: #94a3b8;
        background: #1e293b;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        border: 1px solid #334155;
    }
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
    }
    .status-dot.online {
        background: #22c55e;
        box-shadow: 0 0 8px rgba(34, 197, 94, 0.3);
    }
    .status-dot.offline {
        background: #ef4444;
        box-shadow: 0 0 8px rgba(239, 68, 68, 0.3);
    }

    /* ===== BARRE DE RECHERCHE ===== */
    .search-wrapper {
        background: #1e293b;
        border-radius: 16px;
        padding: 1.5rem 2rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.3);
        border: 1px solid #334155;
        margin-bottom: 1rem;
        transition: all 0.2s ease;
    }
    .search-wrapper:focus-within {
        border-color: #3b82f6;
        box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.1), 0 4px 16px rgba(0,0,0,0.3);
    }
    .search-wrapper .search-input {
        font-size: 1.1rem !important;
        padding: 0.7rem 1rem !important;
        border: none !important;
        background: transparent !important;
        color: #f1f5f9 !important;
    }
    .search-wrapper .search-input::placeholder {
        color: #64748b !important;
    }
    .search-wrapper .search-input:focus {
        box-shadow: none !important;
    }

    /* ===== CHIPS ===== */
    .chips-container {
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
        margin-top: 0.5rem;
    }
    .chip {
        background: #0f172a;
        padding: 0.2rem 0.9rem;
        border-radius: 16px;
        font-size: 0.75rem;
        color: #94a3b8;
        border: 1px solid #334155;
        cursor: pointer;
        transition: all 0.15s ease;
        user-select: none;
    }
    .chip:hover {
        background: #3b82f6;
        color: white;
        border-color: #3b82f6;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.25);
    }

    /* ===== BOUTON RECHERCHE ===== */
    .stButton button[data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, #3b82f6, #2563eb) !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.5rem 2rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 14px rgba(59, 130, 246, 0.3) !important;
        font-size: 0.95rem !important;
    }
    .stButton button[data-testid="baseButton-primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4) !important;
    }
    .stButton button[data-testid="baseButton-primary"]:active {
        transform: translateY(0px) !important;
    }

    /* ===== RÉSULTATS ===== */
    .result-card {
        background: #1e293b;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.6rem;
        border-left: 4px solid #3b82f6;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        border: 1px solid #334155;
        border-left-width: 4px;
        transition: all 0.2s ease;
    }
    .result-card:hover {
        background: #273548;
        box-shadow: 0 4px 16px rgba(0,0,0,0.3);
        transform: translateX(3px);
    }
    .result-title {
        font-weight: 600;
        font-size: 1rem;
        color: #f1f5f9;
        margin-bottom: 0.15rem;
    }
    .result-text {
        color: #94a3b8;
        font-size: 0.85rem;
        line-height: 1.5;
    }
    .result-meta {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        margin-top: 0.4rem;
        flex-wrap: wrap;
    }
    .score-bar-container {
        flex: 1;
        min-width: 60px;
        height: 3px;
        background: #334155;
        border-radius: 3px;
        overflow: hidden;
    }
    .score-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, #3b82f6, #60a5fa);
        border-radius: 3px;
        transition: width 0.3s ease;
    }
    .score-label {
        font-size: 0.65rem;
        font-weight: 500;
        color: #94a3b8;
        min-width: 3rem;
    }
    .doc-id {
        font-size: 0.6rem;
        color: #64748b;
        font-family: monospace;
    }

    /* ===== STATS ===== */
    .stats-bar {
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        padding: 0.4rem 0.8rem;
        background: #1e293b;
        border-radius: 10px;
        margin: 0.5rem 0 1rem 0;
        border: 1px solid #334155;
    }
    .stat-item {
        display: flex;
        align-items: center;
        gap: 0.3rem;
        font-size: 0.8rem;
        color: #94a3b8;
    }
    .stat-item .value {
        color: #f1f5f9;
        font-weight: 600;
    }

    /* ===== HISTORIQUE (SIDEBAR) ===== */
    .history-item {
        display: flex;
        align-items: center;
        gap: 0.3rem;
        padding: 0.3rem 0.5rem;
        background: #0f172a;
        border-radius: 6px;
        font-size: 0.75rem;
        margin-bottom: 0.2rem;
        cursor: pointer;
        transition: all 0.15s ease;
        border: 1px solid #334155;
    }
    .history-item:hover {
        background: #1e293b;
        border-color: #3b82f6;
    }
    .history-time {
        color: #64748b;
        font-size: 0.6rem;
        min-width: 2.4rem;
    }
    .history-query {
        color: #f1f5f9;
        flex: 1;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        font-weight: 500;
    }
    .history-badge {
        font-size: 0.5rem;
        font-weight: 600;
        padding: 0.05rem 0.4rem;
        border-radius: 10px;
        text-transform: uppercase;
        letter-spacing: 0.3px;
    }
    .history-badge.engine {
        background: #1e3a5f;
        color: #60a5fa;
    }
    .history-badge.dataset {
        background: #3b1e3b;
        color: #f472b6;
    }

    /* ===== EXPANDER (RAG) ===== */
    .streamlit-expanderHeader {
        background-color: #1e293b !important;
        color: #f1f5f9 !important;
        border-color: #334155 !important;
        border-radius: 8px !important;
    }
    .streamlit-expanderHeader:hover {
        background-color: #273548 !important;
    }
    .streamlit-expanderContent {
        background-color: #0f172a !important;
        color: #94a3b8 !important;
        border-color: #334155 !important;
        border-radius: 0 0 8px 8px !important;
    }

    /* ===== SUCCESS / ERROR ===== */
    .stAlert {
        background-color: #1e293b !important;
        border-color: #334155 !important;
        color: #f1f5f9 !important;
        border-radius: 10px !important;
    }
    .stAlert .stAlertIcon {
        color: #3b82f6 !important;
    }

    /* ===== FOOTER ===== */
    .footer {
        text-align: center;
        color: #64748b;
        font-size: 0.7rem;
        padding: 1.5rem 0 0.5rem;
        border-top: 1px solid #334155;
        margin-top: 1.5rem;
    }

    /* ===== RESPONSIVE ===== */
    @media (max-width: 640px) {
        .main-header { flex-direction: column; gap: 0.5rem; text-align: center; }
        .search-wrapper { padding: 1rem; }
        .chips-container { justify-content: center; }
        .stats-bar { flex-direction: column; align-items: center; }
        .result-card { padding: 0.8rem 1rem; }
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# EN-TÊTE
# ============================================================
try:
    requests.get("http://localhost:5000", timeout=2)
    api_status = "online"
except:
    api_status = "offline"

status_color = "online" if api_status == "online" else "offline"
status_text = "API" if api_status == "online" else "API hors ligne"

st.markdown(f"""
<div class="main-header">
    <div class="logo">
        <span></span>
        <h1></h1>
        <span class=""></span>
    </div>
    <div class="status">
        <span class="status-dot {status_color}"></span>
        {status_text}
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# INITIALISATION DE L'ÉTAT DE SESSION
# ============================================================
if "history" not in st.session_state:
    st.session_state.history = []
if "current_query" not in st.session_state:
    st.session_state.current_query = ""
if "history_limit" not in st.session_state:
    st.session_state.history_limit = 50

# ============================================================
# BARRE LATÉRALE (TOUTE LA CONFIGURATION)
# ============================================================
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    engine = st.selectbox(
        "Moteur",
        ["BM25", "FAISS", "Hybride", "Rerank", "RAG"],
        index=2,
        help="BM25 = mots-clés · FAISS = sens · Hybride = fusion · Rerank = affiné · RAG = synthèse",
    )

    dataset = st.selectbox(
        "Dataset",
        ["scifact", "nfcorpus", "arguana", "fiqa"],
        index=0,
        help="SciFact (scientifique) · NFCorpus (médical) · FiQA (financier) · Arguana (débat)",
    )

    top_k = st.slider("Top K", 3, 20, 10)

    st.markdown("---")

    api_url = st.text_input("URL de l'API", "http://localhost:5000")

    st.markdown("---")

    st.markdown("### 📜 Historique")
    if st.button("🗑️ Effacer l'historique", use_container_width=True):
        st.session_state.history = []
        st.rerun()

    if st.session_state.history:
        st.caption(f"{len(st.session_state.history)} conversations")
        for idx, item in enumerate(st.session_state.history):
            label = f"🕐 {item['timestamp']} – {item['query'][:25]}{'…' if len(item['query']) > 25 else ''}"
            if st.button(label, key=f"hist_{idx}", use_container_width=True):
                st.session_state.current_query = item["query"]
                st.rerun()
    else:
        st.info("Aucun historique")

    st.markdown("---")

    st.markdown("### 📊 Cache FAISS")
    if st.button("🗑️ Vider le cache", use_container_width=True):
        try:
            resp = requests.post(f"{api_url.rstrip('/')}/cache/clear", timeout=10)
            if resp.status_code == 200:
                st.success("Cache vidé")
            else:
                st.error(f"Erreur {resp.status_code}")
        except Exception as e:
            st.error(f"Erreur : {e}")

    try:
        resp = requests.get(f"{api_url.rstrip('/')}/cache/stats", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("cache"):
                stats = data["cache"]
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("💾 Taille", f"{stats['cache_size']}/{stats['max_cache_size']}")
                    st.metric("✅ Hits", stats['cache_hits'])
                with c2:
                    hit_rate = stats['hit_rate'] * 100
                    st.metric("🎯 Taux", f"{hit_rate:.1f}%")
                    st.metric("❌ Misses", stats['cache_misses'])
                st.caption(f"Cache activé : {'✅' if stats['cache_enabled'] else '❌'}")
        else:
            st.caption("⏳ En attente de l'API...")
    except:
        st.caption("⏳ En attente de l'API...")

# ============================================================
# ZONE PRINCIPALE : RECHERCHE + RÉSULTATS
# ============================================================
with st.container():
    # ===== TEXTE EXPLICATIF DANS LA ZONE JAUNE =====
    st.markdown("""
    <div style="
        background: #1e293b;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin-bottom: 1rem;
        border: 1px solid #334155;
        color: #94a3b8;
        font-size: 0.9rem;
        line-height: 1.6;
    ">
        <strong style="color: #f1f5f9;">Explorez 4 domaines de recherche :</strong><br>
        <span style="color: #60a5fa;">• Sciences</span> (SciFact) ·
        <span style="color: #34d399;">• Médecine</span> (NFCorpus) ·
        <span style="color: #fbbf24;">• Finance</span> (FiQA) ·
        <span style="color: #f472b6;">• Débat</span> (Arguana)
        <br>
    </div>
    """, unsafe_allow_html=True)

    query = st.text_input(
        "Posez votre question",
        value=st.session_state.current_query,
        placeholder="Ex: Quels sont les effets de la caféine sur le sommeil ?",
        label_visibility="collapsed",
        key="search_input",
    )
    if query != st.session_state.current_query:
        st.session_state.current_query = query

    examples = [
        ("☕ Caféine et sommeil", "Quels sont les effets de la caféine sur le sommeil ?"),
        ("🩺 Prévenir le diabète", "Comment prévenir le diabète de type 2 ?"),
        ("💰 Compte d'épargne", "Qu'est-ce qu'un compte d'épargne ?"),
        ("⚖️ Peine de mort", "La peine de mort est-elle justifiée ?"),
    ]

    cols = st.columns(len(examples))
    for i, (label, text) in enumerate(examples):
        with cols[i]:
            if st.button(label, key=f"chip_{i}", use_container_width=True):
                st.session_state.current_query = text
                st.rerun()

    col_btn_left, col_btn_center, col_btn_right = st.columns([2, 2, 2])
    with col_btn_center:
        search_btn = st.button(" Rechercher", type="primary", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# TRAITEMENT DE LA RECHERCHE
# ============================================================
ENDPOINTS = {
    "BM25": "/search_bm25",
    "FAISS": "/search_faiss",
    "Hybride": "/search_hybrid",
    "Rerank": "/search_rerank",
    "RAG": "/search_rag",
}

if search_btn:
    if not query.strip():
        st.warning(" Veuillez saisir une question.")
    else:
        with st.spinner("Recherche en cours…"):
            try:
                start_time = time.time()
                response = requests.post(
                    f"{api_url.rstrip('/')}{ENDPOINTS[engine]}",
                    json={"query": query, "top_k": top_k, "dataset": dataset},
                    timeout=300,
                )
                elapsed = time.time() - start_time

                if response.status_code == 200:
                    data = response.json()

                    history_entry = {
                        "timestamp": datetime.now().strftime("%H:%M"),
                        "query": query,
                        "engine": engine,
                        "dataset": dataset,
                    }
                    st.session_state.history.insert(0, history_entry)
                    if len(st.session_state.history) > st.session_state.history_limit:
                        st.session_state.history.pop()

                    if engine == "RAG":
                        st.success(f"✅ Réponse générée en {data.get('time_ms', 0):.0f} ms")
                        st.markdown("### 📝 Réponse")
                        st.write(data.get("response", "Aucune réponse"))
                        st.markdown("---")
                        st.markdown("### 📚 Sources")
                        for i, doc in enumerate(data.get("sources", []), 1):
                            with st.expander(f"Source {i} – {doc.get('doc_id', '')}"):
                                st.markdown(f"**{doc.get('title', '')}**")
                                st.write(doc.get("text", ""))
                    else:
                        results = data.get("results", [])
                        total = data.get("total_results", 0)
                        time_ms = data.get("time_ms", 0)

                        st.markdown(f"""
                        <div class="stats-bar">
                            <span class="stat-item">🔍 <span class="value">{engine}</span></span>
                            <span class="stat-item">📄 <span class="value">{total}</span> résultats</span>
                            <span class="stat-item">⏱️ <span class="value">{time_ms:.0f}</span> ms</span>
                            <span class="stat-item">📊 Top <span class="value">{top_k}</span></span>
                        </div>
                        """, unsafe_allow_html=True)

                        # ============================================================
                        # AFFICHAGE DES RÉSULTATS AVEC EXPANDER 
                        # ============================================================
                        for i, doc in enumerate(results, 1):
                            score = doc.get("score", 0)
                            title = doc.get("title", "Sans titre")
                            text = doc.get("text", "")
                            doc_id = doc.get("doc_id", "")

                            with st.expander(f"{i}. {title}"):
                                st.markdown(f'<div style="color: #e2e8f0; font-size: 0.9rem; line-height: 1.6; margin-bottom: 0.5rem;">{text}</div>', unsafe_allow_html=True)
                                st.markdown(f"""
                                <div style="display: flex; gap: 1rem; flex-wrap: wrap; color: #94a3b8; font-size: 0.8rem;">
                                    <span style="color: #60a5fa; font-weight: 600;">⭐ Score : {score:.4f}</span>
                                    <span style="color: #64748b; font-family: monospace;">ID : {doc_id}</span>
                                </div>
                                """, unsafe_allow_html=True)

                else:
                    st.error(f" Erreur API {response.status_code} : {response.text}")

            except requests.exceptions.RequestException as e:
                st.error(f" Impossible de joindre l'API : {e}")


# PIED DE PAGE

st.markdown("""
<div class="footer">
     <strong>DocuMind</strong> · Moteur de recherche multi-domaines<br>
</div>
""", unsafe_allow_html=True)