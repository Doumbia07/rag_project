import streamlit as st
import requests

st.set_page_config(page_title="RAG - Moteur de recherche intelligent", layout="wide")
st.title("🔍 Moteur de recherche intelligent (RAG)")

api_url = st.text_input("URL de l'API Flask", "http://localhost:5000")
query = st.text_input("Posez votre question :", placeholder="Ex: Quels sont les effets de la caféine sur le sommeil ?")

col1, col2, col3 = st.columns(3)
with col1:
    engine = st.selectbox("Moteur", ["BM25", "FAISS", "Hybride", "Rerank", "RAG"])
with col2:
    dataset = st.selectbox("Dataset", ["scifact", "nfcorpus", "arguana", "fiqa"], index=0)
with col3:
    top_k = st.slider("Nombre de résultats", 3, 20, 10)

ENDPOINTS = {
    "BM25": "/search_bm25",
    "FAISS": "/search_faiss",
    "Hybride": "/search_hybrid",
    "Rerank": "/search_rerank",
    "RAG": "/search_rag",
}

if st.button("🔎 Lancer la recherche"):
    if not query.strip():
        st.warning("Veuillez saisir une question avant de lancer la recherche.")
    else:
        with st.spinner("Recherche en cours..."):
            try:
                response = requests.post(
                    f"{api_url.rstrip('/')}{ENDPOINTS[engine]}",
                    json={"query": query, "top_k": top_k, "dataset": dataset},
                    timeout=300,
                )
                if response.status_code == 200:
                    data = response.json()
                    if engine == "RAG":
                        st.success(f"✅ Réponse générée en {data.get('time_ms', 0)} ms")
                        st.markdown("### 📝 Réponse")
                        st.write(data.get("response", ""))
                        st.markdown("---")
                        st.markdown("### 📚 Sources")
                        for i, doc in enumerate(data.get("sources", []), start=1):
                            with st.expander(f"Source {i} - {doc.get('doc_id', '')}"):
                                st.markdown(f"**{doc.get('title', '')}**")
                                st.write(doc.get('text', ""))
                    else:
                        st.success(f"✅ {data.get('total_results', 0)} résultats en {data.get('time_ms', 0)} ms")
                        for i, doc in enumerate(data.get("results", []), start=1):
                            with st.expander(f"📄 {i} - Score: {doc.get('score', 0):.4f}"):
                                st.markdown(f"**{doc.get('title', '')}**")
                                st.write(doc.get('text', ""))
                else:
                    st.error(f"Erreur API {response.status_code} : {response.text}")
            except requests.exceptions.RequestException as exc:
                st.error(f"❌ Impossible de joindre l'API : {exc}")

st.markdown("---")
st.markdown("**Conseil :** Lancez l'API Flask avec `python -m backend.app` avant d'utiliser cette interface.")
