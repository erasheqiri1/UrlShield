import html

import streamlit as st

from src.core import available_models, load_metrics, load_model, predict, url_segments
from src.ui import (gauge, impact_chart, model_cards, page_header, reasons_list, reasons_of, segment_notes,
                    votes_of)

cached_model = st.cache_resource(load_model)


def centered_link(url: str):
    st.html(f'<div style="text-align:center;overflow-wrap:anywhere"><code>{html.escape(url)}</code></div>')


page_header("Krahaso dy linke",
            "Shih krah për krah si i vlerëson modeli dy adresa, për shembull një faqe të vërtetë dhe imitimin e saj.")

metrics = load_metrics()
models = available_models()
if not metrics or not models:
    st.warning("Nuk ka model të trajnuar ende.")
    st.stop()

best = metrics["best_model"]
with st.container(key="card_form"):
    model_name = st.selectbox("Modeli", models, index=models.index(best) if best in models else 0,
                              key="compare_model")
    col_a, col_b = st.columns(2)
    url_a = col_a.text_input("Linku i parë", value="https://paypal.com", key="compare_a").strip()
    url_b = col_b.text_input("Linku i dytë", value="https://paypa1.com", key="compare_b").strip()
    _, center, _ = st.columns([2, 1, 2])
    compare = center.button("Krahaso", type="primary", use_container_width=True)

if compare:
    invalid = [u for u in (url_a, url_b) if not u or " " in u or "." not in u]
    if invalid:
        st.error("Shkruaj dy adresa të sakta. Shembull: https://www.google.com")
        st.stop()
    model = cached_model(model_name)
    notes = max(sum(bool(s["reason"]) for s in url_segments(u)) for u in (url_a, url_b))
    gauge_height = 315 + 30 * notes
    for col, key, url in zip(st.columns(2), ("card_gauge_a", "card_gauge_b"), (url_a, url_b)):
        with col.container(key=key, height=gauge_height, border=False):
            proba = predict(model, url)
            gauge(proba, url=url)
            segment_notes(url)

    st.subheader("Vota e modeleve")
    all_models = {name: cached_model(name) for name in models}
    for url in (url_a, url_b):
        st.write(f"`{url}`")
        model_cards(votes_of(all_models, url))

    reference = metrics.get("legit_reference", {})
    rows_a, rows_b = reasons_of(model, url_a, reference), reasons_of(model, url_b, reference)

    st.subheader("Pse doli ky rezultat")
    box_height = 44 * max(len(rows_a), len(rows_b), 1) + 75
    for col, key, url, rows in zip(st.columns(2), ("card_reason_a", "card_reason_b"), (url_a, url_b), (rows_a, rows_b)):
        with col.container(key=key, height=box_height, border=False):
            centered_link(url)
            reasons_list(rows)

    st.subheader("Sa ndikoi secila")
    both = rows_a + rows_b
    x_max = max((abs(r["impact"]) for r in both), default=1) * 1.1
    n_rows = max(len(rows_a), len(rows_b), 1)
    for col, key, url, rows in zip(st.columns(2), ("card_chart_a", "card_chart_b"), (url_a, url_b), (rows_a, rows_b)):
        with col.container(key=key, height=36 * n_rows + 235, border=False):
            centered_link(url)
            impact_chart(rows, x_max=x_max, n_rows=n_rows)
