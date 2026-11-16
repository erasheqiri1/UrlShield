from pathlib import Path

import pandas as pd
import streamlit as st

from src.core import THRESHOLD, available_models, load_metrics, load_model, parse_labels, predict
from src.ui import (colored_url, gauge, model_cards, page_header, reasons_list, reasons_of, verdict_of,
                    votes_of)

cached_model = st.cache_resource(load_model)
QUIZ_FILE = Path(__file__).resolve().parent.parent / "data" / "fresh_test_urls.csv"

page_header("Kuiz: phishing apo jo?", "Shiko linkun, zgjidh përgjigjen dhe krahasohu me modelin. Mos i hap linket.")

metrics = load_metrics()
if not metrics or not QUIZ_FILE.exists():
    st.warning("Kuizi ka nevojë për modelin e trajnuar dhe për skedarin `data/fresh_test_urls.csv`.")
    st.stop()


@st.cache_data
def load_quiz():
    df = pd.read_csv(QUIZ_FILE).dropna(subset=["url", "label"])
    return df.assign(y=parse_labels(df["label"]))[["url", "y"]]


def next_question():
    quiz = load_quiz()
    st.session_state["quiz_row"] = quiz.sample(1).iloc[0].to_dict()
    st.session_state["quiz_answer"] = None


def answer(is_phishing: bool):
    st.session_state["quiz_answer"] = is_phishing


st.session_state.setdefault("quiz_score", {"you": 0, "model": 0, "n": 0})
if "quiz_row" not in st.session_state:
    next_question()

best = metrics["best_model"]
model = cached_model(best)
row = st.session_state["quiz_row"]
score = st.session_state["quiz_score"]

c = st.columns(3)
c[0].metric("Pyetje", score["n"])
c[1].metric("Ti", score["you"])
c[2].metric(f"Modeli ({best})", score["model"])

question = st.container(key="card_question")
question.subheader("A është ky link phishing?")
question.code(row["url"], language=None, wrap_lines=True)

if st.session_state["quiz_answer"] is None:
    a, b = question.columns(2)
    a.button("⚠️ Phishing", on_click=answer, args=(True,), use_container_width=True)
    b.button("✅ Legjitim", on_click=answer, args=(False,), use_container_width=True)
    st.stop()

truth = bool(row["y"])
proba = predict(model, row["url"])
model_says = proba >= THRESHOLD
you_right = st.session_state["quiz_answer"] == truth
model_right = model_says == truth
if not row.get("counted"):
    score["n"] += 1
    score["you"] += int(you_right)
    score["model"] += int(model_right)
    row["counted"] = True
    st.rerun()

st.write(f"Përgjigjja në listë: **{'phishing' if truth else 'legjitim'}**")
(st.success if you_right else st.error)("Ti e gjete saktë." if you_right else "Ti gabove.")
(st.success if model_right else st.error)("Modeli e gjeti saktë." if model_right else "Modeli gaboi.")
with st.container(key="card_result"):
    left, right = st.columns([1, 2])
    with left:
        gauge(proba, verdict_of(proba)[0])
    with right:
        colored_url(row["url"])
        reasons_list(reasons_of(model, row["url"], metrics.get("legit_reference", {})))

votes = votes_of({name: cached_model(name) for name in available_models()}, row["url"])
n_phishing = sum(p >= THRESHOLD for _, p in votes)
st.subheader(f"Vota e modeleve: {n_phishing} nga {len(votes)} thonë phishing")
model_cards(votes)

_, center, _ = st.columns([2, 1, 2])
center.button("Pyetja tjetër", type="primary", on_click=next_question, use_container_width=True)
