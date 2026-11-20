import json
import math
import re
import sqlite3
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from src.features import (BINARY_FEATURES, BRANDS, FEATURE_LABELS, HOSTING_PLATFORMS, is_official_domain,
                          MODEL_FEATURES, SHORTENERS, SUSPICIOUS_TLDS, SUSPICIOUS_WORDS, TEXT_FEATURE,
                          _is_ip, extract_features, features_dataframe, local_brand_in, lookalike_brand,
                          registered_domain, url_domain, url_key, url_text)

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DB_PATH = ROOT / "data" / "history.db"
THRESHOLD = 0.5
UNCERTAIN_LOW, UNCERTAIN_HIGH = 0.35, 0.65
PHISHING_WORDS = {"phishing", "1", "bad", "malicious", "phish"}
LEGIT_WORDS = {"legjitime", "legjitim", "0", "good", "benign", "legit", "legitimate"}
IMBALANCE_RATIO = 1.5
MODEL_NAMES = ["Logistic Regression", "Decision Tree", "Random Forest",
               "Gradient Boosting", "XGBoost"]


def slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def load_metrics():
    p = MODELS_DIR / "metrics.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def load_fresh_metrics():
    p = MODELS_DIR / "fresh_metrics.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def load_model(name: str):
    p = MODELS_DIR / f"{slug(name)}.pkl"
    return joblib.load(p) if p.exists() else None


def available_models():
    return [n for n in MODEL_NAMES if (MODELS_DIR / f"{slug(n)}.pkl").exists()]


@lru_cache(maxsize=1)
def _text_model(modified: float):
    return joblib.load(MODELS_DIR / "text_model.pkl")


@lru_cache(maxsize=1)
def _seen_urls(modified: float):
    return joblib.load(MODELS_DIR / "train_urls.pkl")


def load_seen_urls() -> dict:
    p = MODELS_DIR / "train_urls.pkl"
    return _seen_urls(p.stat().st_mtime) if p.exists() else {"urls": set(), "domains": set()}


def model_frame(urls) -> pd.DataFrame:
    urls = list(urls)
    p = MODELS_DIR / "text_model.pkl"
    scores = _text_model(p.stat().st_mtime).predict_proba([url_text(u) for u in urls])[:, 1]
    return features_dataframe(urls).assign(**{TEXT_FEATURE: scores.round(4)})[MODEL_FEATURES]


def predict(model, url: str) -> float:
    return float(model.predict_proba(model_frame([url]))[0, 1])


def explain(model, url: str, legit_reference: dict, top: int = 6):
    def log_odds(X):
        p = min(max(float(model.predict_proba(X)[0, 1]), 1e-4), 1 - 1e-4)
        return math.log(p / (1 - p))

    base_X = model_frame([url])
    feats = base_X.iloc[0].to_dict()
    base = log_odds(base_X)

    rows = []
    for f in MODEL_FEATURES:
        ref = legit_reference.get(f, feats[f])
        if feats[f] == ref:
            continue
        X = base_X.copy()
        X[f] = ref
        delta = base - log_odds(X)
        if abs(delta) >= 0.05:
            rows.append({"feature": f, "label": FEATURE_LABELS[f],
                         "value": format_value(f, feats[f]), "impact": round(delta, 4)})
    rows.sort(key=lambda r: abs(r["impact"]), reverse=True)
    return rows[:top]


def format_value(feature, value):
    if feature in BINARY_FEATURES:
        return "Po" if value else "Jo"
    if feature == TEXT_FEATURE:
        return f"{value:.0%}"
    return int(value) if float(value).is_integer() else value


def features_table(url: str) -> pd.DataFrame:
    feats = model_frame([url]).iloc[0].to_dict()
    return pd.DataFrame(
        [{"Veçoria": FEATURE_LABELS[k], "Vlera": str(format_value(k, v))} for k, v in feats.items()]
    )


def _host_parts(url: str):
    reg = url_domain(url)
    main = reg.split(".")[0] if reg and not _is_ip(reg) else ""
    return main, lookalike_brand(main)


def reason_text(feature: str, value, url: str) -> str:
    main, brand = _host_parts(url)
    brand = brand or main
    tld = url_domain(url).rsplit(".", 1)[-1]
    binary = {
        "has_ip": ("Përdor adresë IP në vend të domenit", "Përdor domen, jo adresë IP"),
        "has_at": ("Ka simbolin @ në adresë", "Nuk ka simbolin @"),
        "is_https": ("Përdor HTTPS", "Nuk përdor HTTPS"),
        "brand_not_in_domain": ("Emri i një marke është në URL, por domeni nuk është i asaj marke",
                                "Nuk ka emër marke jashtë domenit kryesor"),
        "suspicious_tld": (f"Mbaresa .{tld} përdoret shpesh te phishing",
                           f"Mbaresa .{tld} nuk është në listën e të dyshimtave"),
        "is_shortener": ("Është link i shkurtuar, që e fsheh adresën e vërtetë", "Nuk është link i shkurtuar"),
        "brand_is_domain": (f"Domeni kryesor është marka e njohur {main}", "Domeni kryesor nuk është markë e njohur"),
        "brand_lookalike": (f"Domeni i ngjan markës {brand}, por nuk është ajo", "Domeni nuk imiton markë të njohur"),
        "hosting_platform": ("Është në platformë ku kushdo mund të hapë faqe", "Nuk është në platformë të hapur"),
    }
    if feature in binary:
        return binary[feature][0 if value else 1]
    numeric = {
        "url_length": f"Gjatësia e URL-së: {value:.0f} shenja",
        "domain_length": f"Gjatësia e domenit: {value:.0f} shenja",
        "num_dots": f"Pika në domen: {value:.0f}",
        "num_subdomains": f"Nën-domene: {value:.0f}",
        "num_hyphens": f"Viza (-) në domen: {value:.0f}",
        "num_suspicious_words": f"Fjalë të dyshimta në URL: {value:.0f}",
        "digit_ratio": f"Shifrat përbëjnë {value:.0%} të URL-së",
        "entropy": f"Rastësia e shkronjave (entropia): {value:.2f}",
        "path_depth": f"Pjesë pas domenit: {value:.0f}",
        "suspicious_in_domain": f"Fjalë të dyshimta në domen: {value:.0f}",
        TEXT_FEATURE: f"Teksti i URL-së i ngjan phishing-ut me {value:.0%}",
    }
    return numeric[feature]


def url_segments(url: str) -> list:
    url = url.strip()
    match = re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url)
    scheme = match.group(0) if match else ""
    rest = url[len(scheme):]
    end = re.search(r"[/?#]", rest)
    authority, tail = (rest[:end.start()], rest[end.start():]) if end else (rest, "")
    user, at, host_port = authority.rpartition("@")
    host, colon, port = host_port.partition(":")

    segments = []
    if scheme:
        secure = scheme.lower() == "https://"
        segments.append((scheme, "good" if secure else "risk", "" if secure else "Pa HTTPS"))
    if at:
        segments.append((user + at, "danger", "Simboli @ mund ta fshehë adresën e vërtetë"))

    lower = host.lower()
    bare = lower.removeprefix("www.")
    www = host[:len(lower) - len(bare)]
    if www:
        segments.append((www, "none", ""))
    host = host[len(www):]
    reg = registered_domain(bare)
    if _is_ip(bare):
        segments.append((host, "danger", "Adresë IP në vend të domenit"))
    else:
        sub = host[:len(bare) - len(reg)]
        reg_text = host[len(sub):]
        main = reg.split(".")[0]
        suffix = reg_text[len(main):]
        if sub:
            word = next((w for w in (*BRANDS, *SUSPICIOUS_WORDS) if w in sub.lower()), None)
            if word in BRANDS:
                segments.append((sub, "danger", f"Marka {word} është te nën-domeni, jo te domeni"))
            elif word:
                segments.append((sub, "warn", "Fjalë e dyshimtë"))
            else:
                segments.append((sub, "none", ""))
        imitated = lookalike_brand(main)
        in_main = next((b for b in BRANDS if b in main and b != main), None)
        if not is_official_domain(reg) and not in_main:
            in_main = local_brand_in(main, main)
        word = next((w for w in SUSPICIOUS_WORDS if w in main), None)
        on_platform = any(bare == h or bare.endswith("." + h) for h in HOSTING_PLATFORMS)
        if on_platform:
            segments.append((reg_text, "warn", "Platformë ku kushdo mund të hapë faqe"))
        elif reg in SHORTENERS:
            segments.append((reg_text, "warn", "Link i shkurtuar"))
        else:
            if main in BRANDS and not is_official_domain(reg) and not on_platform:
                segments.append((reg_text[:len(main)], "danger", "Nuk është domeni zyrtar i markës"))
            elif imitated:
                segments.append((reg_text[:len(main)], "danger", f"I ngjan markës {imitated}"))
            elif in_main:
                segments.append((reg_text[:len(main)], "danger", f"Përdor emrin e markës {in_main}"))
            elif word:
                segments.append((reg_text[:len(main)], "warn", "Fjalë e dyshimtë"))
            elif is_official_domain(reg):
                segments.append((reg_text[:len(main)], "good", ""))
            else:
                segments.append((reg_text[:len(main)], "none", ""))
            tld = reg.rsplit(".", 1)[-1]
            if tld in SUSPICIOUS_TLDS:
                segments.append((suffix, "risk", "Mbaresë e dyshimtë"))
            else:
                segments.append((suffix, "none", ""))
    if colon:
        segments.append((colon + port, "none", ""))

    pattern = "|".join(re.escape(w) for w in sorted((*SUSPICIOUS_WORDS, *BRANDS), key=len, reverse=True))
    position = 0
    for found in re.finditer(pattern, tail, flags=re.IGNORECASE):
        if found.start() > position:
            segments.append((tail[position:found.start()], "none", ""))
        word = found.group(0).lower()
        if word in BRANDS:
            segments.append((found.group(0), "danger", f"Marka {word} është pas domenit"))
        else:
            segments.append((found.group(0), "warn", "Fjalë e dyshimtë"))
        position = found.end()
    if position < len(tail):
        segments.append((tail[position:], "none", ""))
    return [{"text": t, "level": level, "reason": r} for t, level, r in segments if t]


def read_uploaded_csv(file) -> pd.DataFrame:
    header = file.readline()
    file.seek(0)
    sep = ";" if b";" in header and b"," not in header else ","
    return pd.read_csv(file, sep=sep)


def parse_labels(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(PHISHING_WORDS).astype(int)


def known_labels(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(PHISHING_WORDS | LEGIT_WORDS)


def imbalance_warning(n_phishing: int, n_legit: int):
    if min(n_phishing, n_legit) == 0:
        return (f"Lista ka vetëm një palë ({n_phishing} phishing, {n_legit} legjitime), "
                "prandaj nota nuk tregon sa mirë i dallon modeli të dyja.")
    if max(n_phishing, n_legit) / min(n_phishing, n_legit) > IMBALANCE_RATIO:
        return (f"Palët nuk janë të barabarta ({n_phishing} phishing, {n_legit} legjitime). "
                "Aplikacioni nuk fshin asgjë për këtë arsye, por nota duhet lexuar me kujdes.")
    return None


def evaluate_on(df: pd.DataFrame, model_names=None) -> dict:
    from src.train import compute_metrics

    n_input = len(df)
    df = df.dropna(subset=["url", "label"]).copy()
    df["url"] = df["url"].astype(str).str.strip()
    n_empty = n_input - len(df)
    valid = known_labels(df["label"])
    n_unknown = int((~valid).sum())
    df = df[valid]
    df["key"] = df["url"].map(url_key)
    before = len(df)
    df = df.drop_duplicates("key")
    n_duplicates = before - len(df)
    seen = load_seen_urls()
    is_seen = df["key"].isin(seen["urls"])
    n_seen = int(is_seen.sum())
    df = df[~is_seen]
    if df.empty:
        raise ValueError("Nuk mbeti asnjë link për testim pasi u hoqën rreshtat e paqartë, "
                         "të përsëritur dhe ata që modeli i ka parë.")
    df["known_domain"] = df["url"].map(url_domain).isin(seen["domains"])
    n_known_domain = int(df["known_domain"].sum())
    df["y"] = parse_labels(df["label"])
    X = model_frame(df["url"])
    results = {}
    for name in model_names or available_models():
        model = load_model(name)
        if model is None:
            continue
        pred = model.predict(X)
        proba = model.predict_proba(X)[:, 1]
        m = compute_metrics(df["y"], pred)
        known = df["known_domain"].to_numpy()
        m["by_domain"] = {label: dict(compute_metrics(df["y"][mask], pred[mask]), n=int(mask.sum()))
                          for label, mask in (("Domen i njohur", known), ("Domen i ri", ~known)) if mask.any()}
        m["rows"] = [
            {"url": u, "true": "phishing" if t else "legjitime",
             "pred": "phishing" if p else "legjitime", "proba": round(float(pr), 4)}
            for u, t, p, pr in zip(df["url"], df["y"], pred, proba)
        ]
        results[name] = m
    return {
        "created": datetime.now().isoformat(timespec="seconds"),
        "n_input": int(n_input),
        "n_empty": int(n_empty),
        "n_unknown_labels": n_unknown,
        "n_duplicates": int(n_duplicates),
        "n_seen_removed": n_seen,
        "n_known_domain": n_known_domain,
        "n_total": int(len(df)),
        "n_phishing": int(df["y"].sum()),
        "n_legit": int((1 - df["y"]).sum()),
        "models": results,
    }


_COMMON_TLDS = ("com|net|org|info|biz|xyz|top|site|online|shop|click|link|live|icu|buzz|app|dev|io|co|me|"
                "al|eu|de|uk|fr|it|ch|at|nl|ru|cn|tk|ml|ga|cf|gq|ly|to|cc")
_URL_PATTERN = re.compile(
    rf"(?:https?://[^\s<>\"']+|www\.[^\s<>\"']+"
    rf"|(?<![@\w.-])(?:[a-z0-9-]+\.)+(?:{_COMMON_TLDS})(?![a-z0-9-])(?:[/?#][^\s<>\"']*)?)",
    flags=re.IGNORECASE,
)


def extract_urls(text: str) -> list:
    urls = []
    for found in _URL_PATTERN.findall(text or ""):
        found = found.rstrip(".,;:!?)]}»\u201d\u2019")
        if found and found not in urls:
            urls.append(found)
    return urls


_ATTACK_SWAPS = (("l", "1"), ("o", "0"), ("e", "3"), ("s", "5"), ("a", "4"), ("i", "1"), ("m", "rn"), ("w", "vv"))


def attack_variants(url: str) -> list:
    domain = url_domain(url)
    main, _, tld = domain.partition(".")
    if not main or not tld:
        return []
    variants = []
    for real, fake in _ATTACK_SWAPS:
        if real in main:
            variants.append(("Shkronjë e ngjashme", f"https://{main.replace(real, fake, 1)}.{tld}/login"))
    if len(main) > 3:
        variants.append(("Shkronjë e hequr", f"https://{main[:-2]}{main[-1]}.{tld}/login"))
        variants.append(("Shkronja të ndërruara", f"https://{main[0]}{main[2]}{main[1]}{main[3:]}.{tld}/login"))
    variants += [
        ("Shkronjë e shtuar", f"https://{main}{main[-1]}.{tld}/login"),
        ("Shifër në fund", f"https://{main}1.{tld}/login"),
        ("Fjalë shtesë", f"https://{main}-secure.{tld}/login"),
        ("Fjalë shtesë", f"https://{main}-login.{tld}"),
        ("Mbaresë tjetër", f"https://{main}.xyz/login"),
        ("Mbaresë tjetër", f"https://{main}.net/login"),
        ("Marka te nën-domeni", f"https://{domain}.secure-check.top/signin"),
        ("Marka te nën-domeni", f"https://{main}.account-verify.com/signin"),
        ("Marka pas domenit", f"https://secure-check.com/{main}/login"),
        ("Platformë e hapur", f"https://{main}-login.pages.dev/"),
        ("Pa HTTPS", f"http://{main}-login.{tld}/verify"),
        ("Adresë IP", f"http://193.42.11.8/{main}/login"),
        ("Simboli @", f"https://{domain}@secure-check.top/login"),
    ]
    seen, unique = {domain}, []
    for kind, variant in variants:
        if variant not in seen:
            seen.add(variant)
            unique.append((kind, variant))
    return unique


def save_fresh_metrics(result: dict, source_name: str):
    result = dict(result, source_file=source_name)
    (MODELS_DIR / "fresh_metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


def _conn():
    DB_PATH.parent.mkdir(exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS scans(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created TEXT NOT NULL,
        url TEXT NOT NULL,
        result TEXT NOT NULL,
        probability REAL NOT NULL,
        model TEXT NOT NULL)""")
    return con


def save_scan(url, probability, model_name):
    result = "Phishing" if probability >= THRESHOLD else "Legjitime"
    with _conn() as con:
        con.execute("INSERT INTO scans(created,url,result,probability,model) VALUES(?,?,?,?,?)",
                    (datetime.now().isoformat(sep=" ", timespec="seconds"), url, result,
                     round(probability, 4), model_name))


def load_history() -> pd.DataFrame:
    with _conn() as con:
        return pd.read_sql_query(
            "SELECT created AS Data, url AS URL, result AS Rezultati, "
            "probability AS \"Probabiliteti për phishing\", model AS Modeli FROM scans ORDER BY id DESC", con)


def clear_history():
    with _conn() as con:
        con.execute("DELETE FROM scans")
