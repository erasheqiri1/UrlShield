import altair as alt
import pandas as pd
import streamlit as st

from src.core import THRESHOLD, attack_variants, available_models, load_metrics, load_model, model_frame
from src.ui import page_header, palette

cached_model = st.cache_resource(load_model)

page_header("Testi i mashtrimit", "Shkruaj një faqe të vërtetë. Programi krijon imitime të saj dhe tregon cilat "
                                  "i kap secili model dhe cilat i shpëtojnë. Asnjë link nuk hapet.")

metrics = load_metrics()
models = available_models()
if not metrics or not models:
    st.warning("Nuk ka model të trajnuar ende.")
    st.stop()

with st.container(key="card_form"):
    url = st.text_input("Faqja e vërtetë", value="https://paypal.com", key="attack_url").strip()
    _, center, _ = st.columns([2, 1.3, 2])
    attack = center.button("Krijo imitimet dhe testo", type="primary", use_container_width=True)

if attack:
    variants = attack_variants(url)
    if not variants:
        st.error("Shkruaj një domen të saktë. Shembull: https://paypal.com")
        st.stop()
    X = model_frame([v for _, v in variants])
    table = pd.DataFrame(variants, columns=["Lloji", "Imitimi"])
    for name in models:
        table[name] = cached_model(name).predict_proba(X)[:, 1]

    summary = pd.DataFrame([{"Modeli": name, "Kapur": float((table[name] >= THRESHOLD).mean())}
                            for name in models]).sort_values("Kapur", ascending=False)
    n_variants = len(table)
    st.subheader(f"Të kapura nga {n_variants} imitime")
    for col, name in zip(st.columns(len(models)), models):
        caught = int((table[name] >= THRESHOLD).sum())
        col.html(f'<div style="border:1px solid #0D9488;background:var(--us-panel);border-radius:14px;'
                 f'padding:12px 8px;text-align:center">'
                 f'<div style="font-size:.85rem;min-height:1.4em">{name}</div>'
                 f'<div style="font-size:1.5rem;font-weight:700;margin:4px 0">{caught} nga {n_variants}</div>'
                 f'<div style="color:var(--us-accent);font-weight:700;font-size:.9rem">'
                 f'Të shpëtuara: {n_variants - caught}</div></div>')

    chart_box = st.container(key="card_chart")
    chart_box.subheader("Sa imitime kapi secili model")
    chart_box.altair_chart(alt.Chart(summary).mark_bar(color=palette()["accent"]).encode(
        x=alt.X("Kapur:Q", title="Pjesa e kapur", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("Modeli:N", sort=None, title=None, axis=alt.Axis(labelLimit=200, labelOverlap=False)),
        tooltip=["Modeli", alt.Tooltip("Kapur:Q", format=".0%")],
    ).properties(height=36 * len(summary) + 80, background="transparent"), use_container_width=True)

    kinds_box = st.container(key="card_kinds")
    kinds_box.subheader("Sa imitime kapi secili model, sipas llojit")
    by_kind = (table[models] >= THRESHOLD).groupby(table["Lloji"]).sum()
    by_kind.insert(0, "Gjithsej", table.groupby("Lloji").size())
    weakest_first = by_kind[models].sum(axis=1).div(by_kind["Gjithsej"]).sort_values().index
    kinds_box.dataframe(by_kind.loc[weakest_first].reset_index(), hide_index=True, use_container_width=True)

    all_box = st.container(key="card_all")
    all_box.subheader("Të gjitha imitimet: probabiliteti për phishing sipas modelit")
    all_box.dataframe(table, hide_index=True, use_container_width=True, column_config={
        name: st.column_config.ProgressColumn(name, min_value=0.0, max_value=1.0, format="percent")
        for name in models})
    all_box.caption("Imitimet janë krijuar nga programi, prandaj kjo provë tregon fortësinë ndaj këtyre "
               "teknikave, jo notën e modelit në realitet.")
