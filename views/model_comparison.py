import json

import pandas as pd
import streamlit as st

from src.core import MODELS_DIR, load_metrics
from src.features import FEATURE_LABELS
from src.ui import confusion_heatmap, grouped_bars, importance_bars, page_header, roc_chart

page_header("Krahasimi i modeleve")

meta = load_metrics()
if not meta:
    st.warning("Nuk ka rezultate. Nise trajnimin: `python -m src.train --data <dataset.csv>`")
    st.stop()

models = meta["models"]
best = meta["best_model"]
bm = models[best]

c = st.columns(5)
c[0].metric("Modeli më i mirë (CV F1)", best)
c[1].metric("F1 në testim", f"{bm['f1']:.4f}")
c[2].metric("Recall", f"{bm['recall']:.4f}")
c[3].metric("ROC-AUC", f"{bm['roc_auc']:.4f}")
c[4].metric("URL në testim", f"{meta['n_test']:,}")

table = pd.DataFrame([
    {"Modeli": n, "Accuracy": m["accuracy"], "Precision": m["precision"], "Recall": m["recall"],
     "F1": m["f1"], "F1 pa URL sintetike": m.get("real_only", {}).get("f1"), "ROC-AUC": m["roc_auc"],
     "Koha e trajnimit (s)": m["train_time_s"]}
    for n, m in models.items()
]).sort_values("F1", ascending=False)
with st.container(key="card_table"):
    st.dataframe(table, hide_index=True, use_container_width=True)
    grouped_bars(table, "Modeli", ["Accuracy", "Precision", "Recall", "F1"])
    st.caption("URL-të sintetike (markat e gjeneruara) janë të lehta për modelin, prandaj nota pa to "
               "tregon më mirë sjelljen me linke reale.")

experiments_path = MODELS_DIR / "experiments.json"
if experiments_path.exists():
    experiments = json.loads(experiments_path.read_text(encoding="utf-8"))
    exp_box = st.container(key="card_experiments")
    exp_box.subheader("Eksperimentet: çka kontribuon secila pjesë")
    exp_box.dataframe(pd.DataFrame(experiments["rows"]), hide_index=True, use_container_width=True)

cv_box = st.container(key="card_cv")
cv_box.subheader(f"Cross-validation me {meta['cv_folds']} ndarje")
cv_box.dataframe(pd.DataFrame([
    {"Modeli": n, "F1 mesatar": m["cv"]["f1_mean"], "F1 ±": m["cv"]["f1_std"],
     "ROC-AUC mesatar": m["cv"]["roc_auc_mean"], "ROC-AUC ±": m["cv"]["roc_auc_std"]}
    for n, m in models.items()
]).sort_values("F1 mesatar", ascending=False), hide_index=True, use_container_width=True)

details = st.container(key="card_details")
choice = details.selectbox("Modeli për detaje", list(models), index=list(models).index(best))
m = models[choice]
tab_cm, tab_roc, tab_src, tab_fi, tab_params, tab_err = details.tabs(
    ["Confusion matrix", "Lakorja ROC", "Sipas dataset-it", "Rëndësia e veçorive", "Parametrat", "Gabimet"])

with tab_cm:
    cm = m["confusion_matrix"]
    st.subheader(choice)
    confusion_heatmap(cm)
    st.markdown(
        f"- **False positive (FP = {cm['fp']:,})**: faqe legjitime e shënuar si phishing.\n"
        f"- **False negative (FN = {cm['fn']:,})**: phishing që kalon pa u kap. "
        "Për siguri, këta janë më të rrezikshmit."
    )

with tab_roc:
    st.subheader("Lakorja ROC")
    roc_chart(models, choice)
    st.caption("Sa më afër këndit lart majtas, aq më mirë. Vija me ndërprerje është hamendësimi i "
               "rastësishëm (AUC 0.5).")

with tab_src:
    st.subheader(f"{choice}: nota për secilin dataset në testim")
    st.dataframe(pd.DataFrame([
        {"Dataset": source, "URL": s["n"], "Accuracy": s["accuracy"], "Precision": s["precision"],
         "Recall": s["recall"], "F1": s["f1"]}
        for source, s in m["by_source"].items()
    ]), hide_index=True, use_container_width=True)
    st.caption("Një notë e përgjithshme mund ta fshehë dobësinë te një burim i caktuar, prandaj "
               "shihen veç e veç. URL-të e gjeneruara kanë modele të ngjashme, prandaj nota e tyre "
               "nuk tregon sjelljen në realitet.")

with tab_fi:
    fi = pd.DataFrame(
        [{"Veçoria": FEATURE_LABELS[k], "Rëndësia": v} for k, v in m["feature_importance"].items()]
    ).sort_values("Rëndësia", ascending=False)
    st.subheader(choice)
    importance_bars(fi)

with tab_params:
    st.subheader(f"{choice}: parametrat e zgjedhur")
    if meta.get("tuned") and m["params"]:
        st.json(m["params"])
        st.caption("I zgjodhi GridSearch me 3 ndarje sipas domenit, vetëm në pjesën e trajnimit.")
    else:
        st.write("Ky trajnim përdori parametrat fillestarë (pa opsionin --tune).")

with tab_err:
    st.subheader(f"Shembuj gabimesh – {choice}")
    if m["errors"]:
        st.dataframe(pd.DataFrame(m["errors"]).rename(
            columns={"url": "URL", "true": "E vërtetë", "pred": "Parashikimi"}),
            hide_index=True, use_container_width=True)
    else:
        st.write("Ky model nuk ka gabime në grupin e testimit.")
