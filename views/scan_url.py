from pathlib import Path

import streamlit as st

from src.core import available_models, features_table, load_metrics, load_model, predict, save_scan
from src.features import has_scheme
from src.report import build_pdf
from src.ui import (disagreement_warning, features_list, impact_chart, model_cards,
                    reasons_list, reasons_of, result_banner, url_notes, verdict_of, votes_of)

cached_model = st.cache_resource(load_model)
HERO = str(Path(__file__).resolve().parent.parent / "assets" / "hero_globi.png")

EXAMPLES = {
    "Google": ("https://www.google.com", ":material/search:"),
    "GitHub login": ("https://github.com/login", ":material/code:"),
    "Imitim i PayPal": ("http://paypa1-login.xyz/verify", ":material/payments:"),
    "Imitim i Instagram": ("https://instagram1.com/login", ":material/photo_camera:"),
    "Adresë IP": ("http://193.42.11.8/bank/verify/index.php", ":material/public:"),
}


def fill_example(url: str):
    st.session_state["url_input"] = url


hero_text, hero_image = st.columns([3, 2], vertical_alignment="center")
with hero_text.container(key="hero_text"):
    st.markdown('# Skano një <span class="hero_accent">URL</span>', unsafe_allow_html=True)
    st.caption("Shkruaj URL-në për të zbuluar nëse është legjitime apo phishing përmes Machine Learning. "
               "Analizohet vetëm teksti i URL-së, pa hapur faqen. Rezultati është parashikim, jo garanci absolute.")
hero_image.image(HERO, use_container_width=True)

metrics = load_metrics()
models = available_models()
if not metrics or not models:
    st.warning("Nuk ka model të trajnuar ende. Nise trajnimin në terminal:\n\n"
               "`python -m src.train --data data/PhiUSIIL_Phishing_URL_Dataset.csv`")
    st.stop()

best = metrics["best_model"]
with st.container(key="card_scan"):
    col_in, col_model = st.columns([3, 1])
    with col_in:
        url = st.text_input(":material/link: URL", placeholder="https://shembull.com/login", key="url_input")
    with col_model:
        model_name = st.selectbox(":material/stacks: Modeli", models,
                                  index=models.index(best) if best in models else 0, key="model_select")

    st.caption("Shembuj: kliko për ta mbushur adresën")
    for col, (label, (example, icon)) in zip(st.columns(len(EXAMPLES)), EXAMPLES.items()):
        col.button(label, key=f"example_{label}", on_click=fill_example, args=(example,), icon=icon,
                   use_container_width=True)

    _, center, _ = st.columns([2, 1, 2])
    analyze = center.button("Analizo", type="primary", key="analyze_btn", icon=":material/play_arrow:",
                            use_container_width=True)

if analyze:
    url = url.strip()
    if not url:
        st.error("Shkruaj një URL para se ta analizosh.")
        st.stop()
    if " " in url or "." not in url:
        st.error("Kjo nuk duket si URL. Shembull i saktë: https://www.google.com")
        st.stop()
    if not has_scheme(url):
        st.info("Adresa s'ka http:// ose https://, prandaj analiza supozon **https://**.")

    model = cached_model(model_name)
    proba = predict(model, url)
    save_scan(url, proba, model_name)

    verdict, _ = verdict_of(proba)
    result_banner(url, proba, model_name)
    url_notes(url)
    if verdict == "E PASIGURT":
        st.warning("Ky rezultat është afër kufirit, prandaj kontrolloje vetë adresën.")

    votes = votes_of({name: cached_model(name) for name in models}, url)
    n_phishing = sum(p >= 0.5 for _, p in votes)
    disagreement_warning(model_name, proba, votes)
    st.subheader(f":material/stacks: Vota e modeleve: {n_phishing} nga {len(votes)} thonë phishing")
    model_cards(votes)

    rows = reasons_of(model, url, metrics.get("legit_reference", {}))
    why, chart = st.columns(2)
    panel_height = 36 * len(rows) + 240 if rows else 200
    with why.container(key="card_why", height=panel_height, border=False):
        st.subheader(":material/info: Pse doli ky rezultat")
        reasons_list(rows)
    with chart.container(key="card_chart", height=panel_height, border=False):
        st.subheader(":material/bar_chart: Sa ndikoi secila")
        impact_chart(rows)

    with st.expander("Të gjitha veçoritë e llogaritura"):
        features_list(features_table(url))

    _, pdf_column, _ = st.columns([2, 1, 2])
    pdf_column.download_button(
        "⬇ Shkarko raportin PDF",
        build_pdf(url, model_name, proba, verdict,
                  [(n, p, "Phishing" if p >= 0.5 else "Legjitime") for n, p in votes],
                  [(r["text"], r["impact"] > 0) for r in rows]),
        file_name="urlshield_raporti.pdf", mime="application/pdf", on_click="ignore", key="pdf_btn",
        use_container_width=True,
    )
