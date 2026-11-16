import streamlit as st

from src.core import available_models, extract_urls, load_metrics, load_model, predict
from src.ui import colored_url, gauge, model_cards, page_header, reasons_list, reasons_of, verdict_of, votes_of

cached_model = st.cache_resource(load_model)

page_header("Skano një QR kod", "Ngarko foton e një QR kodi. Aplikacioni e nxjerr linkun si tekst dhe nuk e hap. "
                                "Vendimin e merr modeli.")

metrics = load_metrics()
models = available_models()
if not metrics or not models:
    st.warning("Nuk ka model të trajnuar ende.")
    st.stop()

try:
    import cv2
    import numpy as np
except ImportError:
    st.error("Për QR kodet duhet paketa OpenCV. Instaloje një herë në terminal:\n\n"
             "`pip install --user opencv-python-headless`")
    st.stop()

with st.container(key="card_form"):
    up = st.file_uploader("Fotoja e QR kodit", type=["png", "jpg", "jpeg"], key="qr_upload")
if up is not None:
    image = cv2.imdecode(np.frombuffer(up.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        st.error("Kjo foto nuk mund të lexohet.")
        st.stop()
    text, _, _ = cv2.QRCodeDetector().detectAndDecode(image)
    if not text:
        st.error("Nuk u gjet QR kod. Provo një foto më të qartë dhe më të drejtë.")
        st.stop()

    box = st.container(key="card_result")
    left, right = box.columns([1, 2])
    left.image(up.getvalue(), use_container_width=True)
    urls = extract_urls(text)
    if not urls:
        right.info("QR kodi nuk ka link. Përmbajtja e tij:")
        right.code(text, language=None)
        st.stop()

    url = urls[0]
    best = metrics["best_model"]
    model = cached_model(best)
    proba = predict(model, url)
    with right:
        gauge(proba, verdict_of(proba)[0])
        colored_url(url)
        st.write(f"Modeli **{best}** jep **{proba:.0%}** probabilitet për phishing.")
    st.subheader("Vota e modeleve")
    model_cards(votes_of({name: cached_model(name) for name in models}, url))
    with st.container(key="card_why"):
        st.subheader("Pse doli ky rezultat")
        reasons_list(reasons_of(model, url, metrics.get("legit_reference", {})))
