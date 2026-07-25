# frontend/gradio_app.py
"""
DocuMind — Moteur de recherche RAG multi-domaines
Interface Gradio sobre, configuration a gauche dans un panneau deroulant.
Comportement identique sur toutes les tailles d'ecran (rien de specifique
a une largeur donnee : le panneau se replie en un rail fin avec une icone,
et se redeploie au clic, partout pareil).
"""

import html
from datetime import datetime

import gradio as gr
import requests

# ============================================================
# CONFIGURATION
# ============================================================
API_URL_DEFAULT = "http://localhost:5000"

ENDPOINTS = {
    "BM25": "/search_bm25",
    "FAISS": "/search_faiss",
    "Hybride": "/search_hybrid",
    "Rerank": "/search_rerank",
    "RAG": "/search_rag",
}

DATASET_LABELS = {
    "scifact": "SciFact — scientifique",
    "nfcorpus": "NFCorpus — médical",
    "fiqa": "FiQA — financier",
    "arguana": "Arguana — débat",
}

MAX_HISTORY = 20

# ============================================================
# CSS
# ============================================================
CUSTOM_CSS = """
:root {
    --bg-primary: #0e1013;
    --bg-secondary: #15171c;
    --bg-card: #1a1d23;
    --bg-card-hover: #20232a;
    --border-color: #262a31;
    --text-primary: #e8eaed;
    --text-secondary: #9aa0a8;
    --text-muted: #686e76;
    --accent: #4d79ee;
    --success: #3aa66b;
    --danger: #d1554b;
    --warning: #c98a3a;
    --radius-lg: 12px;
    --radius-md: 9px;
    --radius-sm: 6px;
}

.gradio-container {
    background: var(--bg-primary) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: var(--text-primary) !important;
    max-width: 1280px !important;
    margin: 0 auto !important;
}

/* ---------- Header ---------- */
.dm-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1rem 0.15rem;
    border-bottom: 1px solid var(--border-color);
    margin-bottom: 1.2rem;
    flex-wrap: wrap;
    gap: 0.6rem;
}
.dm-logo { font-size: 1.15rem; font-weight: 600; letter-spacing: -0.01em; color: var(--text-primary); }
.dm-status { display: flex; align-items: center; gap: 0.45rem; font-size: 0.82rem; color: var(--text-secondary); }
.dm-dot { width: 7px; height: 7px; border-radius: 50%; display: inline-block; }
.dm-dot.online { background: var(--success); }
.dm-dot.offline { background: var(--danger); }

/* ---------- Layout : sidebar gauche + contenu principal ---------- */
.dm-layout {
    display: flex !important;
    align-items: flex-start !important;
    gap: 1.5rem !important;
    flex-wrap: wrap !important;
}

#dm-sidebar-panel {
    width: 300px;
    flex-shrink: 0;
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius-lg) !important;
    padding: 1.2rem !important;
    transition: width 0.2s ease, padding 0.2s ease;
    overflow: hidden;
}
#dm-sidebar-panel.dm-collapsed {
    width: 52px;
    padding: 1.2rem 0.7rem !important;
}

.dm-sidebar-topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1rem;
}
.dm-sidebar-title { font-size: 0.9rem; font-weight: 600; color: var(--text-primary); white-space: nowrap; }
#dm-sidebar-panel.dm-collapsed .dm-sidebar-title { display: none; }

.dm-icon-btn {
    background: transparent;
    border: none;
    color: var(--text-secondary);
    cursor: pointer;
    padding: 0.25rem;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: var(--radius-sm);
}
.dm-icon-btn:hover { color: var(--text-primary); background: var(--bg-card); }
.dm-expand-icon { display: none; }
#dm-sidebar-panel.dm-collapsed .dm-collapse-icon { display: none; }
#dm-sidebar-panel.dm-collapsed .dm-expand-icon { display: flex; }

.dm-sidebar-body { }
#dm-sidebar-panel.dm-collapsed .dm-sidebar-body { display: none; }

.dm-sidebar-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--text-muted);
    margin: 1.1rem 0 0.4rem 0;
}
.dm-sidebar-label.first { margin-top: 0; }

#dm-main-col { min-width: 320px; flex: 1 1 480px; }

/* ---------- Search bar ---------- */
.dm-search-row {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius-lg) !important;
    padding: 0.3rem 0.3rem 0.3rem 1rem !important;
    align-items: center !important;
    flex-wrap: wrap !important;
    gap: 0.5rem !important;
    transition: border-color 0.15s ease;
}
.dm-search-row:focus-within { border-color: var(--accent) !important; }
.dm-search-row #dm-query-box textarea {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    padding: 0.6rem 0.2rem !important;
    font-size: 0.98rem !important;
    color: var(--text-primary) !important;
}
#dm-search-btn {
    background: var(--accent) !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
    font-weight: 500 !important;
    color: white !important;
    flex-shrink: 0;
    margin: 0 !important;
    box-shadow: none !important;
}
#dm-search-btn:hover { background: #3d68d8 !important; }

.dm-toolbar-row { align-items: center !important; margin-top: 0.55rem !important; justify-content: flex-end !important; }
.dm-clear-link {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: var(--text-muted) !important;
    font-size: 0.82rem !important;
}
.dm-clear-link:hover { color: var(--text-secondary) !important; }

/* ---------- Stats bar ---------- */
.dm-stats-bar {
    display: flex;
    align-items: center;
    gap: 1.1rem;
    flex-wrap: wrap;
    font-size: 0.84rem;
    color: var(--text-secondary);
    padding: 0.6rem 0.1rem;
}

/* ---------- Result cards ---------- */
.results-grid { display: flex; flex-direction: column; gap: 0.55rem; margin-top: 0.3rem; }

.result-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-md);
    padding: 0.85rem 1rem;
    cursor: pointer;
    transition: border-color 0.15s ease, background 0.15s ease;
}
.result-card:hover { border-color: #34383f; background: var(--bg-card-hover); }
.result-card summary { list-style: none; cursor: pointer; }
.result-card summary::-webkit-details-marker { display: none; }

.card-head { display: flex; align-items: baseline; gap: 0.6rem; flex-wrap: wrap; }
.card-index { font-weight: 600; font-size: 0.78rem; color: var(--text-muted); min-width: 1.4rem; }
.card-title { font-weight: 500; font-size: 0.98rem; color: var(--text-primary); }

.card-meta { display: flex; align-items: center; gap: 0.6rem; margin-top: 0.45rem; flex-wrap: wrap; }
.score-track { flex: 0 0 80px; height: 4px; border-radius: 999px; background: var(--border-color); overflow: hidden; }
.score-fill { height: 100%; border-radius: 999px; }
.card-score, .card-id { font-size: 0.75rem; color: var(--text-muted); font-family: 'SFMono-Regular', Consolas, monospace; }

.card-preview { color: var(--text-secondary); font-size: 0.88rem; margin: 0.5rem 0 0 0; line-height: 1.5; }
.card-body { border-top: 1px solid var(--border-color); margin-top: 0.65rem; padding-top: 0.55rem; }
.card-full { color: var(--text-primary); font-size: 0.9rem; line-height: 1.6; white-space: pre-wrap; }

/* ---------- RAG answer ---------- */
.dm-rag-answer {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-lg);
    padding: 1.1rem 1.2rem;
    font-size: 0.95rem;
    line-height: 1.65;
    color: var(--text-primary);
}
.dm-source-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-md);
    padding: 0.75rem 0.95rem;
    margin-bottom: 0.5rem;
}
.dm-source-tag { font-size: 0.72rem; color: var(--text-muted); font-weight: 600; letter-spacing: 0.03em; }

/* ---------- History ---------- */
.dm-history-item {
    font-size: 0.82rem;
    color: var(--text-secondary);
    padding: 0.5rem 0.1rem;
    border-bottom: 1px solid var(--border-color);
}
.dm-history-item b { color: var(--text-primary); font-weight: 500; }
.dm-empty { color: var(--text-muted); font-size: 0.86rem; text-align: center; padding: 1rem 0; }

.dm-cache-indicator { font-size: 0.78rem; color: var(--text-muted); margin-top: 0.4rem; }

/* ---------- Bottom note ---------- */
.dm-bottom-note {
    text-align: center;
    padding: 1.5rem 0 0.6rem 0;
    color: var(--text-muted);
    border-top: 1px solid var(--border-color);
    margin-top: 2rem;
}
.dm-bottom-note p { margin: 0.15rem 0; font-size: 0.82rem; }

/* ---------- Config : chips moteur + select dataset (sobre, comme la maquette) ---------- */
#dm-engine-radio label {
    background: transparent !important;
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius-sm) !important;
    padding: 0.4rem 0.7rem !important;
    font-size: 0.85rem !important;
}
#dm-engine-radio label:hover { border-color: var(--text-secondary) !important; }
#dm-engine-radio input:checked + span {
    color: var(--bg-primary) !important;
}
#dm-engine-radio label:has(input:checked) {
    background: var(--text-primary) !important;
    border-color: var(--text-primary) !important;
}
"""

HEAD_SCRIPT = """
<script>
function dmToggleSidebar() {
    var panel = document.getElementById('dm-sidebar-panel');
    if (panel) { panel.classList.toggle('dm-collapsed'); }
}
</script>
"""

# ============================================================
# UTILITAIRES
# ============================================================

def _esc(text):
    return html.escape(str(text or ""))


def check_api_status(api_url):
    try:
        r = requests.get(f"{api_url.rstrip('/')}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def get_cache_hit_rate(api_url):
    try:
        r = requests.get(f"{api_url.rstrip('/')}/cache_stats", timeout=3)
        if r.status_code == 200:
            data = r.json()
            rate = data.get("hit_rate")
            if rate is not None:
                return f"Cache hit rate : {rate:.0%}" if rate <= 1 else f"Cache hit rate : {rate:.0f}%"
    except Exception:
        pass
    return ""


def render_status_html(api_url):
    online = check_api_status(api_url)
    dot_class = "online" if online else "offline"
    label = "API en ligne" if online else "API hors ligne"
    return f'<div class="dm-status"><span class="dm-dot {dot_class}"></span>{label}</div>'


def score_bar_html(score):
    try:
        pct = max(0, min(100, round(float(score) * 100)))
    except (TypeError, ValueError):
        pct = 0
    color = "#3aa66b" if pct > 60 else ("#c98a3a" if pct > 30 else "#d1554b")
    return f'<div class="score-track"><div class="score-fill" style="width:{pct}%;background:{color}"></div></div>'


def render_results_html(results):
    if not results:
        return '<div class="dm-empty">Aucun résultat pour cette recherche.</div>'

    cards = []
    for i, doc in enumerate(results, 1):
        score = doc.get("score", 0)
        title = _esc(doc.get("title", "Sans titre"))
        doc_id = _esc(doc.get("doc_id", ""))
        full_text = _esc(doc.get("text", ""))
        preview = full_text[:300] + ("…" if len(full_text) > 300 else "")

        cards.append(f'''
        <details class="result-card">
            <summary>
                <div class="card-head">
                    <span class="card-index">{i:02d}</span>
                    <span class="card-title">{title}</span>
                </div>
                <div class="card-meta">
                    {score_bar_html(score)}
                    <span class="card-score">{float(score):.4f}</span>
                    <span class="card-id">{doc_id}</span>
                </div>
                <p class="card-preview">{preview}</p>
            </summary>
            <div class="card-body">
                <p class="card-full">{full_text}</p>
            </div>
        </details>
        ''')
    return f'<div class="results-grid">{"".join(cards)}</div>'


def render_rag_html(response_text, sources):
    if not response_text:
        return '<div class="dm-empty">Aucune réponse RAG pour le moment.</div>'
    return f'<div class="dm-rag-answer">{_esc(response_text)}</div>'


def render_sources_html(sources):
    if not sources:
        return '<div class="dm-empty">Aucune source pour cette réponse.</div>'
    cards = []
    for i, doc in enumerate(sources, 1):
        title = _esc(doc.get("title", "Sans titre"))
        doc_id = _esc(doc.get("doc_id", ""))
        text = _esc(doc.get("text", ""))[:500]
        cards.append(f'''
        <div class="dm-source-card">
            <div class="dm-source-tag">SOURCE {i:02d} · {doc_id}</div>
            <div class="card-title" style="margin:0.2rem 0;">{title}</div>
            <p class="card-preview" style="margin-top:0.3rem;">{text}…</p>
        </div>
        ''')
    return "".join(cards)


def render_history_html(history):
    if not history:
        return '<div class="dm-empty">Aucun historique pour le moment.</div>'
    items = []
    for entry in history[:MAX_HISTORY]:
        items.append(
            f'<div class="dm-history-item">{entry["time"]} · <b>{_esc(entry["query"][:60])}</b> '
            f'· {entry["engine"]} · {entry["dataset"]}</div>'
        )
    return "".join(items)


# ============================================================
# APPELS API
# ============================================================

def call_api(query, engine, dataset, top_k, api_url):
    try:
        response = requests.post(
            f"{api_url.rstrip('/')}{ENDPOINTS[engine]}",
            json={"query": query.strip(), "top_k": top_k, "dataset": dataset},
            timeout=300,
        )
        if response.status_code == 200:
            return response.json(), None
        return None, f"Erreur API {response.status_code} : {response.text[:200]}"
    except Exception as e:
        return None, f"Impossible de contacter l'API : {e}"


# ============================================================
# INTERFACE GRADIO
# ============================================================

with gr.Blocks(
    title="DocuMind — Moteur de recherche RAG",
) as demo:

    history_state = gr.State([])

    # ---------- Header ----------
    with gr.Row(elem_classes="dm-header"):
        gr.HTML('<div class="dm-logo">DocuMind</div>')
        status_html = gr.HTML(render_status_html(API_URL_DEFAULT))

    # ---------- Corps : panneau de configuration (gauche) + contenu principal ----------
    with gr.Row(elem_classes="dm-layout", equal_height=False):

        # ===== Panneau de configuration =====
        with gr.Column(elem_id="dm-sidebar-panel"):
            gr.HTML('''
                <div class="dm-sidebar-topbar">
                    <span class="dm-sidebar-title">Configuration</span>
                    <button type="button" class="dm-icon-btn dm-collapse-icon" onclick="dmToggleSidebar()" aria-label="Réduire">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M15 18l-6-6 6-6"/></svg>
                    </button>
                    <button type="button" class="dm-icon-btn dm-expand-icon" onclick="dmToggleSidebar()" aria-label="Développer">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 18l6-6-6-6"/></svg>
                    </button>
                </div>
            ''')

            with gr.Column(elem_classes="dm-sidebar-body"):
                gr.HTML('<div class="dm-sidebar-label first">Moteur</div>')
                engine_choice = gr.Radio(
                    choices=list(ENDPOINTS.keys()),
                    value="Hybride",
                    show_label=False,
                    elem_id="dm-engine-radio",
                )

                gr.HTML('<div class="dm-sidebar-label">Dataset</div>')
                dataset_choice = gr.Dropdown(
                    choices=[(v, k) for k, v in DATASET_LABELS.items()],
                    value="scifact",
                    show_label=False,
                )

                gr.HTML('<div class="dm-sidebar-label">Paramètres</div>')
                top_k_slider = gr.Slider(minimum=3, maximum=20, value=10, step=1, label="Top K")
                api_url_input = gr.Textbox(label="URL de l'API", value=API_URL_DEFAULT)
                cache_indicator = gr.HTML("", elem_classes="dm-cache-indicator")

                gr.HTML('<div class="dm-sidebar-label">Historique</div>')
                history_output = gr.HTML('<div class="dm-empty">Aucun historique pour le moment.</div>')

        # ===== Contenu principal =====
        with gr.Column(elem_id="dm-main-col"):
            with gr.Row(elem_classes="dm-search-row"):
                query_input = gr.Textbox(
                    placeholder="Ex : Quels sont les effets de la caféine sur le sommeil ?",
                    show_label=False,
                    lines=1,
                    elem_id="dm-query-box",
                    scale=5,
                    container=False,
                )
                search_btn = gr.Button("Rechercher", variant="primary", elem_id="dm-search-btn", scale=1, min_width=120)

            with gr.Row(elem_classes="dm-toolbar-row"):
                clear_btn = gr.Button("Effacer", elem_classes="dm-clear-link", size="sm", scale=0)

            stats_output = gr.HTML('<div class="dm-stats-bar">En attente d\'une recherche…</div>')

            with gr.Tabs():
                with gr.TabItem("Résultats"):
                    results_output = gr.HTML('<div class="dm-empty">Lancez une recherche pour voir les résultats.</div>')
                with gr.TabItem("Réponse RAG"):
                    rag_output = gr.HTML('<div class="dm-empty">Aucune réponse RAG pour le moment.</div>')
                with gr.TabItem("Sources"):
                    sources_output = gr.HTML('<div class="dm-empty">Aucune source pour le moment.</div>')

    # ---------- Note de bas de page ----------
    gr.HTML('''
    <div class="dm-bottom-note">
        <p><strong>Posez votre question sur n'importe quel sujet</strong></p>
        <p>Scientifique · Médical · Financier · Argumentatif — 5 moteurs de recherche à comparer</p>
    </div>
    ''')

    # ============================================================
    # LOGIQUE
    # ============================================================

    def on_search(query, engine, dataset, top_k, api_url, history):
        if not query or not query.strip():
            warn = '<div class="dm-stats-bar">Veuillez saisir une question.</div>'
            return (
                gr.update(),
                warn,
                gr.update(),
                gr.update(),
                render_history_html(history),
                history,
                "",
            )

        data, error = call_api(query, engine, dataset, top_k, api_url)

        if error:
            stats_html = f'<div class="dm-stats-bar">{_esc(error)}</div>'
            return (
                '<div class="dm-empty">La recherche a échoué.</div>',
                stats_html,
                '<div class="dm-empty">Aucune réponse RAG pour le moment.</div>',
                '<div class="dm-empty">Aucune source pour le moment.</div>',
                render_history_html(history),
                history,
                "",
            )

        cache_html = get_cache_hit_rate(api_url)

        new_history = [{
            "time": datetime.now().strftime("%H:%M"),
            "query": query.strip(),
            "engine": engine,
            "dataset": dataset,
        }] + history
        new_history = new_history[:MAX_HISTORY]

        if engine == "RAG":
            response_text = data.get("response", "")
            sources = data.get("sources", [])
            time_ms = data.get("time_ms", 0)
            stats_html = (
                f'<div class="dm-stats-bar">'
                f'<span>Réponse générée</span>'
                f'<span>{time_ms:.0f} ms</span>'
                f'<span>{len(sources)} sources</span>'
                f'</div>'
            )
            return (
                '<div class="dm-empty">Ce moteur ne retourne pas de liste de résultats — voir l\'onglet "Réponse RAG".</div>',
                stats_html,
                render_rag_html(response_text, sources),
                render_sources_html(sources),
                render_history_html(new_history),
                new_history,
                cache_html,
            )
        else:
            results = data.get("results", [])
            total = data.get("total_results", len(results))
            time_ms = data.get("time_ms", 0)
            stats_html = (
                f'<div class="dm-stats-bar">'
                f'<span>{total} résultats</span>'
                f'<span>{time_ms:.0f} ms</span>'
                f'<span>{engine}</span>'
                f'</div>'
            )
            return (
                render_results_html(results),
                stats_html,
                '<div class="dm-empty">Ce moteur ne génère pas de réponse RAG. Sélectionnez le moteur "RAG".</div>',
                '<div class="dm-empty">Sources disponibles uniquement en mode RAG.</div>',
                render_history_html(new_history),
                new_history,
                cache_html,
            )

    def on_clear():
        return (
            "",
            '<div class="dm-empty">Lancez une recherche pour voir les résultats.</div>',
            '<div class="dm-stats-bar">En attente d\'une recherche…</div>',
            '<div class="dm-empty">Aucune réponse RAG pour le moment.</div>',
            '<div class="dm-empty">Aucune source pour le moment.</div>',
        )

    outputs_list = [results_output, stats_output, rag_output, sources_output, history_output, history_state, cache_indicator]
    inputs_list = [query_input, engine_choice, dataset_choice, top_k_slider, api_url_input, history_state]

    search_btn.click(on_search, inputs=inputs_list, outputs=outputs_list)
    query_input.submit(on_search, inputs=inputs_list, outputs=outputs_list)

    clear_btn.click(
        on_clear,
        inputs=[],
        outputs=[query_input, results_output, stats_output, rag_output, sources_output],
    )

    api_url_input.change(fn=render_status_html, inputs=[api_url_input], outputs=[status_html])

    demo.load(fn=render_status_html, inputs=[api_url_input], outputs=[status_html])
    demo.load(fn=get_cache_hit_rate, inputs=[api_url_input], outputs=[cache_indicator])

# ============================================================
# LANCEMENT
# ============================================================
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        css=CUSTOM_CSS,
        head=HEAD_SCRIPT,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        ),
        footer_links=[],
    )