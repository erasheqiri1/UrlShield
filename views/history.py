import altair as alt
import pandas as pd
import streamlit as st

from src.core import (available_models, clear_history, load_history, load_metrics,
                      load_model, predict, read_uploaded_csv, save_scan)
from src.ui import page_header, palette, results_table

page_header("Historia e skanimeve")

with st.expander("Skano shumë URL njëherësh (CSV me kolonën url)"):
    up = st.file_uploader("Skedari CSV", type="csv", key="batch_upload")
    models = available_models()
    meta = load_metrics()
    if up is not None and models and st.button("Skano të gjitha", type="primary"):
        df = read_uploaded_csv(up)
        if "url" not in df.columns:
            st.error("CSV-ja duhet ta ketë kolonën `url`.")
        else:
            name = meta["best_model"] if meta and meta["best_model"] in models else models[0]
            model = load_model(name)
            urls = df["url"].dropna().astype(str).str.strip().tolist()
            bar = st.progress(0.0)
            for i, u in enumerate(urls, 1):
                save_scan(u, predict(model, u), name)
                bar.progress(i / len(urls))
            st.success(f"U skanuan {len(urls)} URL me {name}.")

hist = load_history()
if hist.empty:
    st.info("Ende nuk ka skanime. Shko te **Skano URL** dhe analizo një adresë.")
    st.stop()

with st.container(key="card_filter"):
    c1, c2 = st.columns([3, 1])
    with c1:
        q = st.text_input("Kërko URL", key="hist_search")
    with c2:
        res = st.selectbox("Rezultati", ["Të gjitha", "Phishing", "Legjitime"], key="hist_filter")

view = hist
if q:
    view = view[view["URL"].str.contains(q, case=False, regex=False)]
if res != "Të gjitha":
    view = view[view["Rezultati"] == res]

c = st.columns(3)
c[0].metric("Gjithsej", len(hist))
c[1].metric("Phishing", int((hist["Rezultati"] == "Phishing").sum()))
c[2].metric("Legjitime", int((hist["Rezultati"] == "Legjitime").sum()))

by_day = hist.assign(Dita=pd.to_datetime(hist["Data"]).dt.strftime("%Y-%m-%d")) \
    .groupby(["Dita", "Rezultati"]).size().reset_index(name="Skanime")
charts = st.container(key="card_chart")
charts.altair_chart(alt.Chart(by_day).mark_bar().encode(
    x=alt.X("Dita:O", title=None, axis=alt.Axis(labelAngle=0)),
    y=alt.Y("Skanime:Q", title="Skanime", axis=alt.Axis(tickMinStep=1)),
    color=alt.Color("Rezultati:N", title=None, legend=alt.Legend(orient="bottom"),
                    scale=alt.Scale(domain=["Phishing", "Legjitime"], range=["#dc2626", palette()["accent"]])),
    tooltip=["Dita", "Rezultati", "Skanime"],
).properties(height=240, background="transparent"), use_container_width=True)

with st.container(key="card_records"):
    results_table(view)

_, d1, d2, _ = st.columns([2, 1, 1, 2])
d1.download_button("⬇ Eksporto CSV", view.to_csv(index=False).encode("utf-8"),
                   file_name="urlshield_history.csv", mime="text/csv", use_container_width=True)
if d2.button("Fshij historinë", use_container_width=True):
    clear_history()
    st.rerun()
