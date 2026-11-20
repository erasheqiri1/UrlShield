import io
import textwrap
from datetime import datetime
from pathlib import Path

LOGO = Path(__file__).resolve().parent.parent / "assets" / "logo.png"
INK, MUTED, EDGE, SOFT = "#081719", "#55706F", "#CFE3E0", "#F0FDFA"
TEAL, BRIGHT, DARK, RED, AMBER = "#0D9488", "#2DD4BF", "#10272B", "#DC2626", "#CA8A04"
THRESHOLD = 0.5
UNCERTAIN = (0.35, 0.65)
WIDTH, HEIGHT, MARGIN = 595, 842, 40


def _figure(url, model_name, proba, verdict, votes, reasons):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.image import imread
    from matplotlib.patches import FancyBboxPatch, Rectangle

    fig = Figure(figsize=(WIDTH / 72, HEIGHT / 72))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(HEIGHT, 0)
    ax.axis("off")

    def card(x, y, w, h, face="white", edge=EDGE, width=1.0, radius=8):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                                    facecolor=face, edgecolor=edge, linewidth=width))

    def text(x, y, value, size=10, color=INK, weight="normal", ha="left", family=None):
        ax.text(x, y, value, fontsize=size, color=color, fontweight=weight, ha=ha, va="center", family=family)

    ax.add_patch(Rectangle((0, 0), WIDTH, 92, facecolor=INK, edgecolor="none"))
    ax.add_patch(Rectangle((0, 90), WIDTH, 2, facecolor=BRIGHT, edgecolor="none"))
    if LOGO.exists():
        logo = fig.add_axes([MARGIN / WIDTH, 1 - 82 / HEIGHT, 64 / WIDTH, 64 / HEIGHT])
        logo.imshow(imread(LOGO))
        logo.axis("off")
    text(112, 37, "URLShield", 21, "white", "bold")
    text(112, 62, "Raporti i skanimit", 11, BRIGHT)
    text(WIDTH - MARGIN, 49, f"{datetime.now():%d.%m.%Y %H:%M}", 10, SOFT, ha="right")

    color = AMBER if UNCERTAIN[0] <= proba <= UNCERTAIN[1] else RED if proba >= THRESHOLD else TEAL
    width = WIDTH - 2 * MARGIN
    y = 112
    card(MARGIN, y, width, 82, face=f"{color}14", edge=color, width=1.4, radius=10)
    text(MARGIN + 18, y + 30, verdict, 19, color, "bold")
    text(MARGIN + 18, y + 57, f"Modeli {model_name} jep {proba:.0%} probabilitet për phishing.", 10, INK)
    text(WIDTH - MARGIN - 18, y + 41, f"{proba:.0%}", 31, color, "bold", ha="right")

    y += 100
    lines = textwrap.wrap(url, 74, break_long_words=True, break_on_hyphens=False) or [url]
    if len(lines) > 4:
        lines = lines[:3] + [lines[3][:71] + "..."]
    h = 40 + 16 * len(lines)
    card(MARGIN, y, width, h)
    text(MARGIN + 16, y + 18, "Adresa", 11, TEAL, "bold")
    for i, part in enumerate(lines):
        text(MARGIN + 16, y + 38 + 16 * i, part, 9.5, INK, family="monospace")

    y += h + 16
    h = 44 + 24 * len(votes)
    card(MARGIN, y, width, h)
    text(MARGIN + 16, y + 18, "Vota e modeleve", 11, TEAL, "bold")
    bar_x, bar_w = MARGIN + 170, 190
    text(bar_x + bar_w + 34, y + 18, "Probabiliteti për phishing", 8.5, MUTED, ha="right")
    for i, (name, value, vote) in enumerate(votes):
        row = y + 44 + 24 * i
        phishing = value >= THRESHOLD
        tone = RED if phishing else TEAL
        text(MARGIN + 16, row, name, 10)
        ax.add_patch(FancyBboxPatch((bar_x, row - 4), bar_w, 8, boxstyle="round,pad=0,rounding_size=4",
                                    facecolor=EDGE, edgecolor="none"))
        if value > 0.005:
            ax.add_patch(FancyBboxPatch((bar_x, row - 4), max(bar_w * value, 8), 8,
                                        boxstyle="round,pad=0,rounding_size=4", facecolor=tone, edgecolor="none"))
        text(bar_x + bar_w + 34, row, f"{value:.0%}", 10, weight="bold", ha="right")
        text(WIDTH - MARGIN - 16, row, vote, 10, tone, "bold", ha="right")

    y += h + 16
    wrapped = [(textwrap.wrap(t, 62) or [t], to_phishing) for t, to_phishing in reasons]
    h = 42 + sum(16 * len(parts) + 8 for parts, _ in wrapped) if wrapped else 66
    card(MARGIN, y, width, h)
    text(MARGIN + 16, y + 18, "Çka ndikoi më së shumti", 11, TEAL, "bold")
    row = y + 42
    if wrapped:
        for parts, to_phishing in wrapped:
            tone = RED if to_phishing else TEAL
            text(MARGIN + 16, row, "Drejt phishing" if to_phishing else "Drejt legjitime", 9.5, tone, "bold")
            for part in parts:
                text(MARGIN + 116, row, part, 10)
                row += 16
            row += 8
    else:
        text(MARGIN + 16, row, "Asnjë veçori nuk e ndryshon dukshëm rezultatin.", 10)

    note = ("Analizë statike e tekstit të URL-së: faqja nuk hapet. Rezultati është parashikim i një "
            "modeli të machine learning, jo garanci absolute.")
    for i, part in enumerate(textwrap.wrap(note, 96)):
        text(WIDTH / 2, HEIGHT - 44 + 13 * i, part, 8.5, MUTED, ha="center")
    return fig


def build_pdf(url: str, model_name: str, proba: float, verdict: str, votes: list, reasons: list) -> bytes:
    from matplotlib.backends.backend_pdf import PdfPages

    fig = _figure(url, model_name, proba, verdict, votes, reasons)
    out = io.BytesIO()
    with PdfPages(out, metadata={"Title": "URLShield – Raporti i skanimit"}) as pdf:
        pdf.savefig(fig)
    return out.getvalue()
