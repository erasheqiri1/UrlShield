from pathlib import Path

import streamlit as st

from src.ui import global_styles

LOGO = str(Path(__file__).resolve().parent / "assets" / "logo.png")

st.set_page_config(page_title="URLShield", page_icon=LOGO, layout="wide")

pages = [
    st.Page("views/scan_url.py", title="Skano URL", icon=":material/search:", default=True),
    st.Page("views/compare_urls.py", title="Krahaso dy linke", icon=":material/compare_arrows:"),
    st.Page("views/scan_message.py", title="Skano një mesazh", icon=":material/mail:"),
    st.Page("views/scan_qr.py", title="Skano QR kod", icon=":material/qr_code_2:"),
    st.Page("views/quiz.py", title="Kuiz: phishing apo jo?", icon=":material/track_changes:"),
    st.Page("views/attack_test.py", title="Testi i mashtrimit", icon=":material/gpp_maybe:"),
    st.Page("views/model_comparison.py", title="Krahasimi i modeleve", icon=":material/bar_chart:"),
    st.Page("views/fresh_test.py", title="Testi me URL të reja", icon=":material/science:"),
    st.Page("views/history.py", title="Historia", icon=":material/history:"),
]
current = st.navigation(pages, position="hidden")

global_styles()

with st.sidebar:
    st.html("""<style>
        [data-testid="stSidebarUserContent"] {padding-top: 0; margin-top: -1.5rem;}
        .st-key-active_page a {background: #2DD4BF; border-radius: 10px;}
        .st-key-active_page a:hover {background: #2DD4BF;}
        .st-key-active_page a * {color: #081719 !important;}
        [data-testid="stPageLink-NavLink"]:hover {background: #1C3B40;}
        .st-key-ml_box {background: #1C3B4066; border: 1px solid #1C3B40; border-radius: 10px;
                        padding: .5rem .75rem; margin-top: .4rem; cursor: default;}
        .st-key-ml_box p {font-size: .8rem; color: #F0FDFA; white-space: nowrap; text-align: center;}
        .st-key-ml_box [data-testid="stIconMaterial"] {color: #2DD4BF;}
    </style>""")
    _, center, _ = st.columns([1, 2.7, 1])
    center.image(LOGO, use_container_width=True)
    st.html('<div style="text-align:center;font-size:1.7rem;font-weight:700;line-height:1.2">URLShield</div>'
            '<div style="text-align:center;font-size:.76rem;color:#F0FDFA;opacity:.8;margin-top:2px">'
            'AI-Powered Phishing URL Detection</div>'
            '<div style="position:relative;height:1px;margin:10px 0 6px;'
            'background:linear-gradient(90deg,transparent,#0D9488 15%,#0D9488 85%,transparent)">'
            '<div style="position:absolute;left:32%;right:32%;top:-1px;height:3px;border-radius:2px;'
            'background:#2DD4BF;box-shadow:0 0 8px 2px #2DD4BF99"></div></div>')
    for page in pages:
        if page.title == current.title:
            with st.container(key="active_page"):
                st.page_link(page)
        else:
            st.page_link(page)
    with st.container(key="ml_box"):
        st.markdown(":material/memory: **Kliko me besim**")

current.run()
