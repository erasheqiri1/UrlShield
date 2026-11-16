import pandas as pd
import streamlit as st

from src.core import THRESHOLD, available_models, extract_urls, load_metrics, load_model, predict
from src.ui import page_header, results_table, verdict_of

cached_model = st.cache_resource(load_model)

EXAMPLE = ("I nderuar klient, llogaria juaj është bllokuar. Verifikoni të dhënat këtu: "
           "http://paypa1-login.xyz/verify ose vizitoni https://github.com/login për ndihmë.")


def fill_example():
    st.session_state["message_text"] = EXAMPLE


page_header("Skano një mesazh", "Ngjit një SMS ose email. Aplikacioni i gjen linket si tekst dhe nuk i hap. "
                                "Vendimin për secilin link e merr modeli.")

metrics = load_metrics()
models = available_models()
if not metrics or not models:
    st.warning("Nuk ka model të trajnuar ende.")
    st.stop()

best = metrics["best_model"]
with st.container(key="card_form"):
    model_name = st.selectbox("Modeli", models, index=models.index(best) if best in models else 0,
                              key="message_model")
    text = st.text_area("Mesazhi", height=170, key="message_text")
    st.button("Mbush me një mesazh shembull", on_click=fill_example)
    _, center, _ = st.columns([2, 1.3, 2])
    analyze = center.button("Gjej dhe analizo linket", type="primary", use_container_width=True)

if analyze:
    urls = extract_urls(text)
    if not urls:
        st.info("Nuk u gjet asnjë link në këtë tekst.")
        st.stop()
    model = cached_model(model_name)
    rows = [{"URL": url, "Rezultati": verdict_of(p)[0], "Probabiliteti": p}
            for url in urls for p in [predict(model, url)]]
    n_phishing = sum(r["Probabiliteti"] >= THRESHOLD for r in rows)
    c = st.columns(2)
    c[0].metric("Linke të gjetura", len(rows))
    c[1].metric("Duken phishing", n_phishing)
    if n_phishing:
        st.error("Ky mesazh ka të paktën një link të dyshimtë. Mos e kliko.")
    with st.container(key="card_results"):
        results_table(pd.DataFrame(rows))
        st.caption("Për detaje, kopjoje linkun te faqja **Skano URL**.")
