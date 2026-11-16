import base64
import html
import math

import altair as alt
import pandas as pd
import streamlit as st

from src.core import (THRESHOLD, UNCERTAIN_HIGH, UNCERTAIN_LOW, explain, predict, reason_text,
                      url_segments)

LEVEL_COLORS = {
    "good": ("#2DD4BF", "Në rregull"),
    "warn": ("#ca8a04", "Kujdes"),
    "risk": ("#ea580c", "E dyshimtë"),
    "danger": ("#dc2626", "Rrezik"),
}
TEAL, TEAL_BRIGHT, PANEL, CARD_EDGE = "#0D9488", "#2DD4BF", "#10272B", "#1C3B40"
CHART_COLORS = ["#0D9488", "#2DD4BF", "#99F6E4", "#F0FDFA"]
ROC_COLORS = ["#2DD4BF", "#F0FDFA", "#0D9488", "#FACC15", "#99F6E4"]
PHISHING_COLOR, LEGIT_COLOR, UNCERTAIN_COLOR = "#dc2626", TEAL, "#ca8a04"


def verdict_of(proba: float) -> tuple:
    if UNCERTAIN_LOW <= proba <= UNCERTAIN_HIGH:
        return "E PASIGURT", UNCERTAIN_COLOR
    if proba >= THRESHOLD:
        return "Duket PHISHING", PHISHING_COLOR
    return "Duket LEGJITIME", LEGIT_COLOR


def _svg_image(svg: str, style: str) -> str:
    data = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f'<img src="data:image/svg+xml;base64,{data}" alt="" style="{style}">'


def page_header(title: str, caption: str = None):
    with st.container(key="page_header"):
        st.title(title)
        if caption:
            st.caption(caption)


def global_styles():
    st.html("""<style>
        [data-testid="stMainBlockContainer"] {padding-top: 3.75rem;}   /* shiriti lart i mbulon 3.75rem */
        [data-testid="stAlertContainer"] {border-radius: 18px;}
        [class*="st-key-card_"], [data-testid="stExpander"] details, [data-testid="stMetric"] {
            background: #10272B; border: 1px solid #0D9488; border-radius: 18px;}
        [class*="st-key-card_"] {padding: 1.25rem 1.5rem;}
        [data-testid="stMetric"] {padding: .85rem 1.1rem;}
        [class*="st-key-card_"] [data-testid="stMetric"] {background: #1C3B40;}
        [class*="st-key-card_"] h3 {text-align: center; padding-top: 0;}
        [data-testid="stExpander"] summary {background: transparent; border-radius: 18px;}
        [data-testid="stExpander"] summary:hover {background: #1C3B40;}
        [data-testid="stMain"] input, [data-testid="stMain"] textarea,
        [data-testid="stMain"] [data-baseweb="select"] > div,
        [data-testid="stFileUploaderDropzone"] {background: #1C3B40; border-radius: 12px;}
        [data-testid="stMain"] button[kind="secondary"] {
            background: #1C3B40; border: 1px solid #0D9488; border-radius: 12px;
            transition: border-color .2s, box-shadow .2s;}
        [data-testid="stMain"] button[kind="secondary"]:hover {
            border-color: #2DD4BF; box-shadow: 0 0 10px #2DD4BF55; color: #F0FDFA;}
        [data-testid="stMain"] button[kind="primary"] {
            background: #0D9488; border: none; border-radius: 16px; min-height: 3.1rem;
            color: #F0FDFA; transition: background .2s;}
        [data-testid="stMain"] button[kind="primary"]:hover {background: #2DD4BF; color: #081719;}
        [data-testid="stMain"] button[kind="primary"]:hover * {color: #081719;}
        [data-testid="stMain"] button[kind="primary"] p {font-size: 1.1rem; font-weight: 600;}
        .st-key-page_header, .st-key-hero_text {border-left: 4px solid #2DD4BF; padding-left: 1rem;}
        .st-key-page_header h1, .st-key-hero_text h1 {padding-top: 0;}
        .st-key-hero_text {margin-top: .75rem;}
        .st-key-hero_text .hero_accent {color: #2DD4BF;}
    </style>""")


def results_table(table: pd.DataFrame, max_height: int = 440):
    bright = {PHISHING_COLOR: "#f87171", UNCERTAIN_COLOR: "#facc15", LEGIT_COLOR: TEAL_BRIGHT}
    head = "".join(f'<th style="text-align:left;padding:10px 12px;font-weight:500;position:sticky;top:0;'
                   f'background:{CARD_EDGE}">{html.escape(col)}</th>' for col in table.columns)
    rows = []
    for record in table.to_dict("records"):
        cells = []
        for col, value in record.items():
            if col == "Probabiliteti":
                color = bright[verdict_of(value)[1]]
                cells.append(f'<td style="padding:9px 12px;min-width:170px"><div style="display:flex;align-items:center;gap:10px">'
                             f'<div style="flex:1;height:8px;border-radius:4px;background:{CARD_EDGE}">'
                             f'<div style="width:{value:.0%};height:8px;border-radius:4px;background:{color}"></div></div>'
                             f'<span style="min-width:40px;text-align:right;color:{color};font-weight:600">{value:.0%}</span>'
                             f'</div></td>')
            else:
                cells.append(f'<td style="padding:9px 12px;overflow-wrap:anywhere">{html.escape(str(value))}</td>')
        rows.append(f'<tr style="border-top:1px solid {CARD_EDGE}">{"".join(cells)}</tr>')
    st.html(f'<div style="max-height:{max_height}px;overflow-y:auto;border-radius:10px">'
            f'<table style="width:100%;border-collapse:collapse"><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def features_list(table: pd.DataFrame):
    head = "".join(f'<th style="text-align:left;padding:10px 12px;font-weight:500;position:sticky;top:0;'
                   f'background:#1C3B40">{html.escape(col)}</th>' for col in table.columns)
    rows = "".join(
        '<tr style="border-top:1px solid #1C3B40">'
        + "".join(f'<td style="padding:9px 12px">{html.escape(str(value))}</td>' for value in row)
        + "</tr>"
        for row in table.itertuples(index=False)
    )
    st.html(f'<div style="max-height:440px;overflow-y:auto;border-radius:10px">'
            f'<table style="width:100%;border-collapse:collapse"><thead>'
            f'<tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>')


def result_banner(url: str, proba: float, model_name: str):
    verdict, color = verdict_of(proba)
    first, _, last = verdict.rpartition(" ")
    bright = {LEGIT_COLOR: TEAL_BRIGHT, PHISHING_COLOR: "#f87171", UNCERTAIN_COLOR: "#facc15"}[color]
    mark = {LEGIT_COLOR: "M8 12.5l3 3 5.5-6", PHISHING_COLOR: "M12 7.5v5.5M12 16.2v.3",
            UNCERTAIN_COLOR: "M9.8 9.3a2.3 2.3 0 1 1 3.4 2.1c-.8.5-1.2 1-1.2 1.9M12 16.2v.3"}[color]
    icon = _svg_image(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
                      f'stroke="{bright}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">'
                      f'<path d="M12 2.5l8 3v6c0 5-3.4 8.4-8 10-4.6-1.6-8-5-8-10v-6z"/><path d="{mark}"/></svg>',
                      "width:58px;height:58px;flex:none")
    parts, _ = _url_parts(url)
    st.html(f'<div style="display:flex;flex-wrap:wrap;align-items:stretch;border:1px solid {color};'
            f'border-radius:18px;background:{PANEL};overflow:hidden">'
            f'<div style="display:flex;align-items:center;gap:14px;padding:16px 22px;flex:1 1 260px">{icon}'
            f'<div style="font-size:1.55rem;font-weight:700;line-height:1.2">{html.escape(first)} '
            f'<span style="color:{bright}">{html.escape(last)}</span></div></div>'
            f'<div style="padding:16px 22px;flex:1 1 280px;min-width:0;border-left:1px solid {CARD_EDGE}">'
            f'<div style="font-weight:600;margin-bottom:6px">Linku i analizuar</div>'
            f'<div style="font-family:monospace;font-size:1rem;overflow-wrap:anywhere">{"".join(parts)}</div></div>'
            f'<div style="padding:16px 22px;flex:0 1 270px;border-left:1px solid {CARD_EDGE}">'
            f'<div style="font-weight:600;margin-bottom:6px">{html.escape(model_name)}</div>'
            f'<div style="background:{CARD_EDGE};border-radius:999px;padding:6px 14px;text-align:center">'
            f'<b>{proba:.0%}</b> probabilitet për phishing</div></div></div>')


def _url_parts(url: str) -> tuple:
    parts, notes = [], []
    for seg in url_segments(url):
        text = html.escape(seg["text"])
        color = LEVEL_COLORS.get(seg["level"], (None,))[0]
        if color:
            parts.append(f'<span style="color:{color};font-weight:700;background:{color}22;'
                         f'border-radius:4px;padding:1px 2px">{text}</span>')
        else:
            parts.append(f"<span>{text}</span>")
        if seg["reason"] and color:
            notes.append(f'<span style="display:inline-block;margin:6px 12px 0 0;font-size:.85rem">'
                         f'<span style="color:{color}">●</span> <code>{text}</code> '
                         f'{html.escape(seg["reason"])}</span>')
    return parts, notes


def url_notes(url: str):
    _, notes = _url_parts(url)
    if notes:
        st.html(f'<div style="background:{PANEL};border:1px solid {TEAL};border-radius:18px;'
                f'padding:8px 18px 14px;text-align:center">{"".join(notes)}</div>')


def colored_url(url: str):
    parts, notes = _url_parts(url)
    st.html(f'<div style="font-family:monospace;font-size:1.15rem;line-height:1.7;'
            f'overflow-wrap:anywhere">{"".join(parts)}</div>'
            f'<div>{"".join(notes)}</div>')


def gauge(proba: float, label: str):
    def point(p, radius):
        angle = math.pi * (1 - p)
        return 110 + radius * math.cos(angle), 110 - radius * math.sin(angle)

    def arc(start, end, color):
        x1, y1 = point(start, 90)
        x2, y2 = point(end, 90)
        return (f'<path d="M {x1:.1f} {y1:.1f} A 90 90 0 0 1 {x2:.1f} {y2:.1f}" fill="none" '
                f'stroke="{color}" stroke-width="18"/>')

    x, y = point(proba, 72)
    _, color = verdict_of(proba)
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 220 135">'
           f'{arc(0, UNCERTAIN_LOW, LEGIT_COLOR)}{arc(UNCERTAIN_LOW, UNCERTAIN_HIGH, UNCERTAIN_COLOR)}'
           f'{arc(UNCERTAIN_HIGH, 1, PHISHING_COLOR)}'
           f'<line x1="110" y1="110" x2="{x:.1f}" y2="{y:.1f}" stroke="#F0FDFA" stroke-width="4" '
           f'stroke-linecap="round"/><circle cx="110" cy="110" r="7" fill="#F0FDFA"/>'
           f'<text x="110" y="133" text-anchor="middle" font-size="20" font-weight="700" '
           f'font-family="sans-serif" fill="{color}">{proba:.0%}</text></svg>')
    st.html(_svg_image(svg, "width:100%;max-width:300px;display:block;margin:auto") +
            f'<div style="text-align:center;font-weight:700;color:{color};font-size:1.05rem">'
            f'{html.escape(label)}</div>')


def model_cards(rows: list):
    for col, (name, proba) in zip(st.columns(len(rows)), rows):
        phishing = proba >= THRESHOLD
        color = PHISHING_COLOR if phishing else TEAL
        vote = "Phishing" if phishing else "Legjitime"
        bright = "#f87171" if phishing else TEAL_BRIGHT
        col.html(f'<div style="border:1px solid {color};background:{PANEL};border-radius:14px;'
                 f'padding:12px 8px;text-align:center;height:100%">'
                 f'<div style="font-size:.85rem;min-height:1.4em">{html.escape(name)}</div>'
                 f'<div style="font-size:1.5rem;font-weight:700;margin:4px 0">{proba:.0%}</div>'
                 f'<div style="color:{bright};font-weight:700">{vote}</div></div>')


def reasons_of(model, url: str, legit_reference: dict) -> list:
    from src.features import BINARY_FEATURES, TEXT_FEATURE, extract_features

    rows = explain(model, url, legit_reference)
    raw = extract_features(url)
    for row in rows:
        f = row["feature"]
        value = row["value"]
        if f == TEXT_FEATURE:
            value = float(str(value).rstrip("%")) / 100
        elif f in BINARY_FEATURES:
            value = raw[f]
        row["text"] = reason_text(f, value, url)
    return rows


def reasons_list(rows: list):
    if not rows:
        st.write("Asnjë veçori e vetme nuk e ndryshon dukshëm rezultatin.")
        return
    st.caption("Këto janë disa nga 20 veçoritë, ato që ndikuan më shumë. Vendimi përfundimtar "
               "del nga të gjitha bashkë.")
    for row in rows:
        icon = ":red[:material/warning:]" if row["impact"] > 0 else ":green[:material/check_circle:]"
        st.markdown(f"{icon} {row['text']}", unsafe_allow_html=False)


def centered_legend(items: list):
    entries = "".join(
        f'<span><span style="display:inline-block;width:12px;height:12px;border-radius:2px;'
        f'background:{color};margin-right:6px"></span>{html.escape(label)}</span>'
        for label, color in items
    )
    st.html(f'<div style="display:flex;justify-content:center;flex-wrap:wrap;gap:16px;font-size:.9rem">{entries}</div>')


def impact_chart(rows: list, x_max: float = None, n_rows: int = None):
    if not rows:
        return
    x_scale = alt.Scale(domain=[-x_max, x_max]) if x_max else alt.Undefined
    data = pd.DataFrame([{"Veçoria": r["label"], "Ndikimi": round(r["impact"], 2),
                          "Drejtimi": "Drejt phishing" if r["impact"] > 0 else "Drejt legjitime"}
                         for r in rows])
    chart = alt.Chart(data).mark_bar().encode(
        x=alt.X("Ndikimi:Q", title="Forca e ndikimit", scale=x_scale),
        y=alt.Y("Veçoria:N", sort=None, title=None, axis=alt.Axis(labelLimit=240, labelOverlap=False)),
        color=alt.Color("Drejtimi:N", legend=None,
                        scale=alt.Scale(domain=["Drejt phishing", "Drejt legjitime"],
                                        range=[PHISHING_COLOR, TEAL_BRIGHT])),
        tooltip=["Veçoria", "Ndikimi", "Drejtimi"],
    ).properties(height=36 * (n_rows or len(data)) + 80,
                 background="transparent")
    st.altair_chart(chart, use_container_width=True)
    centered_legend([("Drejt phishing", PHISHING_COLOR), ("Drejt legjitime", TEAL_BRIGHT)])


def votes_of(models: dict, url: str) -> list:
    return [(name, predict(model, url)) for name, model in models.items()]


def disagreement_warning(model_name: str, proba: float, votes: list):
    n_phishing = sum(p >= THRESHOLD for _, p in votes)
    chosen_phishing = proba >= THRESHOLD
    majority_phishing = n_phishing > len(votes) / 2
    if chosen_phishing != majority_phishing:
        chosen, majority = ("phishing", "legjitime") if chosen_phishing else ("legjitime", "phishing")
        n_majority = n_phishing if majority_phishing else len(votes) - n_phishing
        st.warning(f"**Kujdes:** {model_name} voton \"{chosen}\", por {n_majority} nga "
                   f"{len(votes)} modele votojnë \"{majority}\".")


def grouped_bars(table: pd.DataFrame, id_column: str, value_columns: list):
    data = table.melt(id_vars=id_column, value_vars=value_columns, var_name="Masa", value_name="Vlera")
    chart = alt.Chart(data).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=alt.X(f"{id_column}:N", sort=None, title=None, axis=alt.Axis(labelAngle=0, labelLimit=170)),
        xOffset=alt.XOffset("Masa:N", sort=value_columns),
        y=alt.Y("Vlera:Q", title=None, scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("Masa:N", sort=value_columns, legend=None,
                        scale=alt.Scale(domain=value_columns, range=CHART_COLORS[:len(value_columns)])),
        tooltip=[id_column, "Masa", alt.Tooltip("Vlera:Q", format=".4f")],
    ).properties(height=280, background="transparent")
    st.altair_chart(chart, use_container_width=True)
    centered_legend(list(zip(value_columns, CHART_COLORS)))


def importance_bars(table: pd.DataFrame):
    chart = alt.Chart(table).mark_bar(color=TEAL_BRIGHT, cornerRadiusEnd=3).encode(
        x=alt.X("Rëndësia:Q"),
        y=alt.Y("Veçoria:N", sort=None, title=None, axis=alt.Axis(labelLimit=260, labelOverlap=False)),
        tooltip=["Veçoria", alt.Tooltip("Rëndësia:Q", format=".4f")],
    ).properties(height=26 * len(table) + 50, background="transparent")
    st.altair_chart(chart, use_container_width=True)


def confusion_heatmap(cm: dict):
    data = pd.DataFrame([
        {"E vërtetë": "Legjitime", "Parashikimi": "Legjitime", "Numri": cm["tn"], "Saktë": "Saktë"},
        {"E vërtetë": "Legjitime", "Parashikimi": "Phishing", "Numri": cm["fp"], "Saktë": "Gabim"},
        {"E vërtetë": "Phishing", "Parashikimi": "Legjitime", "Numri": cm["fn"], "Saktë": "Gabim"},
        {"E vërtetë": "Phishing", "Parashikimi": "Phishing", "Numri": cm["tp"], "Saktë": "Saktë"},
    ])
    base = alt.Chart(data).encode(
        x=alt.X("Parashikimi:N", axis=alt.Axis(orient="top", labelAngle=0, labelFontSize=13)),
        y=alt.Y("E vërtetë:N", axis=alt.Axis(labelFontSize=13)),
    )
    cells = base.mark_rect(cornerRadius=6, stroke=TEAL, strokeWidth=1).encode(
        color=alt.Color("Saktë:N", legend=None,
                        scale=alt.Scale(domain=["Saktë", "Gabim"], range=[TEAL, CARD_EDGE])),
        tooltip=["E vërtetë", "Parashikimi", "Numri"],
    )
    labels = base.mark_text(fontSize=24, fontWeight="bold", color="white").encode(
        text=alt.Text("Numri:Q", format=","))
    st.altair_chart((cells + labels).properties(height=280, background="transparent"), use_container_width=True)
    centered_legend([("Saktë", TEAL), ("Gabim", CARD_EDGE)])


def roc_chart(models: dict, highlight: str):
    rows = []
    names = [f"{name} (AUC {m['roc_auc']:.3f})" for name, m in models.items()]
    for name, m in models.items():
        for fpr, tpr in zip(m["roc_curve"]["fpr"], m["roc_curve"]["tpr"]):
            rows.append({"Modeli": f"{name} (AUC {m['roc_auc']:.3f})", "fpr": fpr, "tpr": tpr})
    lines = alt.Chart(pd.DataFrame(rows)).mark_line(strokeWidth=2).encode(
        x=alt.X("fpr:Q", title="Alarme të rreme (False Positive Rate)", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("tpr:Q", title="Phishing i kapur (True Positive Rate)", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("Modeli:N", legend=None, scale=alt.Scale(domain=names, range=ROC_COLORS)),
        tooltip=["Modeli", "fpr", "tpr"],
    )
    diagonal = alt.Chart(pd.DataFrame({"fpr": [0, 1], "tpr": [0, 1]})).mark_line(
        strokeDash=[6, 4], color="gray").encode(x="fpr:Q", y="tpr:Q")
    st.altair_chart((lines + diagonal).properties(height=400, background="transparent"), use_container_width=True)
    centered_legend(list(zip(names, ROC_COLORS)))
