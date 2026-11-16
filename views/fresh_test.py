import pandas as pd
import streamlit as st

from src.core import evaluate_on, imbalance_warning, load_metrics, read_uploaded_csv
from src.ui import grouped_bars, page_header

page_header("Testi me URL të reja", "Sa mirë punon modeli me URL që nuk i ka parë kurrë, jashtë dataset-it të trajnimit.")

with st.container(key="card_form"):
    st.markdown("Ngarko një CSV me kolonat `url` dhe `label`. Kolona `label` pranon `phishing` / `legjitime` "
                "(ose `1` / `0`).")
    up = st.file_uploader("Skedari CSV", type="csv", key="fresh_upload")
    _, center, _ = st.columns([2, 1.3, 2])
    run = center.button("Testo modelet", type="primary", use_container_width=True, disabled=up is None)

if up is None:
    st.session_state.pop("fresh_result", None)
elif run:
    df = read_uploaded_csv(up)
    if not {"url", "label"} <= set(df.columns):
        st.error("CSV-ja duhet t'i ketë kolonat `url` dhe `label`.")
        st.stop()
    try:
        with st.spinner("Duke testuar…"):
            st.session_state["fresh_result"] = dict(evaluate_on(df), source_file=up.name)
    except ValueError as err:
        st.error(str(err))
        st.stop()

fresh = st.session_state.get("fresh_result")
meta = load_metrics()
if not fresh:
    st.info("Ngarko një skedar CSV dhe kliko **Testo modelet** për të parë rezultatet.")
    st.stop()

st.caption(f"Burimi: {fresh['source_file']} · {fresh['n_total']} URL "
           f"({fresh['n_phishing']} phishing, {fresh['n_legit']} legjitime)")

removed = [(fresh.get("n_empty", 0), "rreshta bosh"),
           (fresh.get("n_unknown_labels", 0), "me `label` të panjohur"),
           (fresh.get("n_duplicates", 0), "linke të përsëritura"),
           (fresh.get("n_seen_removed", 0), "linke që modeli i ka parë gjatë trajnimit")]
removed = [f"**{n}** {text}" for n, text in removed if n]
if removed:
    st.info(f"Nga {fresh['n_input']} rreshta u hoqën nga testi: " + ", ".join(removed) + ".")
if fresh.get("n_known_domain"):
    st.warning(f"**{fresh['n_known_domain']}** nga {fresh['n_total']} linke janë në domene që dalin "
               "edhe në trajnim. Linket janë të reja, por modeli i ka parë më parë domenet e tyre, "
               "prandaj i ka më të lehta. Për këtë arsye nota mund të dalë pak më e lartë se në realitet.")
warning = imbalance_warning(fresh["n_phishing"], fresh["n_legit"])
if warning:
    st.warning(warning)

rows = []
for name, m in fresh["models"].items():
    base = meta["models"].get(name, {}) if meta else {}
    rows.append({"Modeli": name,
                 "F1 – grupi i testimit": base.get("f1"),
                 "F1 – URL të reja": m["f1"],
                 "Ndryshimi": round(m["f1"] - base["f1"], 4) if base else None,
                 "Precision (reja)": m["precision"], "Recall (reja)": m["recall"]})
comp = pd.DataFrame(rows)
summary = st.container(key="card_summary")
summary.dataframe(comp, hide_index=True, use_container_width=True)
with summary:
    grouped_bars(comp, "Modeli", ["F1 – grupi i testimit", "F1 – URL të reja"])

by_domain = [{"Modeli": name, "Domeni": kind, "URL": s["n"], "Precision": s["precision"],
              "Recall": s["recall"], "F1": s["f1"]}
             for name, m in fresh["models"].items() for kind, s in m.get("by_domain", {}).items()]
if by_domain:
    domain_box = st.container(key="card_domains")
    domain_box.subheader("URL të reja në domene të njohura dhe të reja")
    domain_box.dataframe(pd.DataFrame(by_domain), hide_index=True, use_container_width=True)
    domain_box.caption("\"Domen i njohur\" do të thotë që linku është i ri, por modeli e ka parë domenin e tij "
                       "gjatë trajnimit. \"Domen i ri\" do të thotë që modeli nuk e ka parë kurrë as linkun, "
                       "as domenin. Rreshti \"Domen i ri\" tregon më mirë sa punon modeli në jetën reale.")

box = st.container(key="card_detail")
box.subheader("Rezultatet për secilën URL")
model = box.selectbox("Modeli", list(fresh["models"]), key="fresh_model")
only_wrong = box.checkbox("Shfaq vetëm gabimet", value=True, key="only_wrong")
detail = pd.DataFrame(fresh["models"][model]["rows"])
detail["Saktë"] = detail["true"] == detail["pred"]
if only_wrong:
    detail = detail[~detail["Saktë"]]
box.dataframe(detail.rename(columns={"url": "URL", "true": "E vërtetë", "pred": "Parashikimi",
                                     "proba": "Prob. phishing"}),
              hide_index=True, use_container_width=True)
