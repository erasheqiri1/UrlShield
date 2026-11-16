import ipaddress
import math
import re
from collections import Counter
from urllib.parse import urlsplit


SUSPICIOUS_WORDS = [
    "login", "log-in", "signin", "sign-in", "verify", "verification", "secure",
    "account", "update", "confirm", "banking", "bank", "password", "webscr",
    "wallet", "suspend", "unlock", "billing", "invoice", "support", "authenticate",
]

BRANDS = [
    "paypal", "apple", "icloud", "microsoft", "office365", "outlook", "google",
    "gmail", "facebook", "instagram", "whatsapp", "amazon", "netflix", "ebay",
    "dhl", "fedex", "binance", "coinbase", "linkedin", "dropbox", "adobe",
    "github", "youtube", "twitter", "yahoo", "telegram", "tiktok", "spotify", "discord",
]

BRAND_DOMAINS = {f"{brand}.com" for brand in BRANDS} | {
    "telegram.org", "telegram.me", "t.me", "office.com", "microsoft365.com",
    "amazon.co.uk", "amazon.de", "amazon.fr", "amazon.it", "amazon.es", "amazon.ca", "amazon.co.jp",
    "amazon.com.au", "amazon.in", "amazon.com.br", "amazon.com.mx", "amazon.nl", "amazon.se", "amazon.pl",
    "amazon.com.tr", "amazon.sg", "amazon.ae", "amazon.sa",
    "ebay.co.uk", "ebay.de", "ebay.fr", "ebay.it", "ebay.es", "ebay.ca", "ebay.com.au", "ebay.at",
    "ebay.ch", "ebay.ie", "ebay.pl", "ebay.nl", "ebay.be",
    "yahoo.co.jp", "yahoo.co.uk", "dhl.de", "apple.co", "youtu.be", "discord.gg",
} | set("""google.com google.ad google.ae google.com.af google.com.ag google.al google.am google.co.ao google.com.ar
google.as google.at google.com.au google.az google.ba google.com.bd google.be google.bf google.bg google.com.bh google.bi
google.bj google.com.bn google.com.bo google.com.br google.bs google.bt google.co.bw google.by google.com.bz google.ca
google.cd google.cf google.cg google.ch google.ci google.co.ck google.cl google.cm google.cn google.com.co google.co.cr
google.com.cu google.cv google.com.cy google.cz google.de google.dj google.dk google.dm google.com.do google.dz
google.com.ec google.ee google.com.eg google.es google.com.et google.fi google.com.fj google.fm google.fr google.ga
google.ge google.gg google.com.gh google.com.gi google.gl google.gm google.gr google.com.gt google.gy google.com.hk
google.hn google.hr google.ht google.hu google.co.id google.ie google.co.il google.im google.co.in google.iq google.is
google.it google.je google.com.jm google.jo google.co.jp google.co.ke google.com.kh google.ki google.kg google.co.kr
google.com.kw google.kz google.la google.com.lb google.li google.lk google.co.ls google.lt google.lu google.lv
google.com.ly google.co.ma google.md google.me google.mg google.mk google.ml google.com.mm google.mn google.com.mt
google.mu google.mv google.mw google.com.mx google.com.my google.co.mz google.com.na google.com.ng google.com.ni
google.ne google.nl google.no google.com.np google.nr google.nu google.co.nz google.com.om google.com.pa google.com.pe
google.com.pg google.com.ph google.com.pk google.pl google.pn google.com.pr google.ps google.pt google.com.py
google.com.qa google.ro google.ru google.rw google.com.sa google.com.sb google.sc google.se google.com.sg google.sh
google.si google.sk google.com.sl google.sn google.so google.sm google.sr google.st google.com.sv google.td google.tg
google.co.th google.com.tj google.tl google.tm google.tn google.to google.com.tr google.tt google.com.tw google.co.tz
google.com.ua google.co.ug google.co.uk google.com.uy google.co.uz google.com.vc google.co.ve google.co.vi google.com.vn
google.vu google.ws google.rs google.co.za google.co.zm google.co.zw google.cat""".split())

LOCAL_DOMAINS = {
    "raiffeisen-kosovo.com": "raiffeisen", "procreditbank-kos.com": "procredit",
    "teb-kos.com": "teb", "nlb-kos.com": "nlb", "bkt-ks.com": "bkt", "bekonomike.com": "bekonomike",
    "bpbbank.com": "bpb", "bqk-kos.org": "bqk", "kosovotelecom.com": "kosovotelecom", "ipko.com": "ipko",
    "bkt.com.al": "bkt", "tiranabank.al": "tiranabank",
}
LOCAL_BRANDS = sorted(set(LOCAL_DOMAINS.values()))
LOCAL_DOMAIN_NAMES = {reg.split(".")[0]: brand for reg, brand in LOCAL_DOMAINS.items()}
SHORT_LOCAL_BRANDS = {b for b in LOCAL_BRANDS if len(b) <= 4}

SUSPICIOUS_TLDS = {
    "xyz", "top", "tk", "ml", "ga", "cf", "gq", "zip", "mov", "click", "link",
    "work", "rest", "fit", "buzz", "cam", "icu", "monster", "support", "live",
}

SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly",
    "cutt.ly", "rebrand.ly", "shorturl.at", "tiny.cc", "rb.gy",
}

HOSTING_PLATFORMS = (
    "sites.google.com", "docs.google.com", "forms.gle", "github.io", "blogspot.com", "weebly.com",
    "wixsite.com", "firebaseapp.com", "web.app", "pages.dev", "netlify.app", "vercel.app",
    "herokuapp.com", "amazonaws.com", "000webhostapp.com", "godaddysites.com", "sharepoint.com",
)

LOOKALIKES = (("rn", "m"), ("vv", "w"), ("0", "o"), ("1", "l"), ("3", "e"), ("5", "s"))
CONFUSABLES = str.maketrans("аеорсухіјѕԁһӏԛԝкмптвαοιυνρκτεχ", "aeopcyxijsdhlqwkmntbaoiuvpktex")

SECOND_LEVEL = {"co", "com", "net", "org", "gov", "edu", "ac", "ltd", "plc", "gob", "gouv", "mil", "nhs",
                "sch", "nic", "or", "ne", "go"}

FEATURE_NAMES = [
    "url_length",
    "domain_length",
    "num_dots",
    "num_subdomains",
    "num_hyphens",
    "has_ip",
    "has_at",
    "is_https",
    "num_suspicious_words",
    "brand_not_in_domain",
    "suspicious_tld",
    "digit_ratio",
    "entropy",
    "path_depth",
    "is_shortener",
    "brand_is_domain",
    "suspicious_in_domain",
    "brand_lookalike",
    "hosting_platform",
]

FEATURE_LABELS = {
    "url_length": "Gjatësia e URL-së",
    "domain_length": "Gjatësia e domenit",
    "num_dots": "Numri i pikave në domen",
    "num_subdomains": "Numri i nën-domeneve",
    "num_hyphens": "Numri i vizave (-) në domen",
    "has_ip": "Ka IP në vend të domenit",
    "has_at": "Ka simbolin @",
    "is_https": "Përdor HTTPS",
    "num_suspicious_words": "Fjalë të dyshimta",
    "brand_not_in_domain": "Emër marke jashtë domenit kryesor",
    "suspicious_tld": "TLD e dyshimtë",
    "digit_ratio": "Raporti i shifrave",
    "entropy": "Entropia (Shannon)",
    "path_depth": "Thellësia e path-it",
    "is_shortener": "Shkurtues URL-je",
    "brand_is_domain": "Domeni kryesor është markë e njohur",
    "suspicious_in_domain": "Fjalë të dyshimta në domen",
    "brand_lookalike": "Domeni imiton një markë",
    "hosting_platform": "Platformë ku kushdo publikon",
    "text_score": "Teksti i URL-së i ngjan phishing-ut",
}

TEXT_FEATURE = "text_score"
MODEL_FEATURES = FEATURE_NAMES + [TEXT_FEATURE]

BINARY_FEATURES = {"has_ip", "has_at", "is_https", "brand_not_in_domain",
                   "suspicious_tld", "is_shortener", "brand_is_domain",
                   "brand_lookalike", "hosting_platform"}


def shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _normalize(url: str) -> str:
    url = re.sub(r"^([a-zA-Z][a-zA-Z0-9+.-]*://)?www\.", r"\1", url, flags=re.IGNORECASE)
    if re.fullmatch(r"([a-zA-Z][a-zA-Z0-9+.-]*://)?[^/?#]+/", url):
        url = url[:-1]
    return url


def _parse(url: str):
    url = _normalize(url.strip())
    has_scheme = bool(re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url))
    if not has_scheme:
        url = "https://" + url
    try:
        parts = urlsplit(url)
        host = (parts.hostname or "").lower()
    except ValueError:
        parts = urlsplit(re.sub(r"[\[\]]", "", url))
        host = (parts.hostname or "").lower()
    return url, host, parts.path or "", has_scheme


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return bool(re.fullmatch(r"(0x[0-9a-f]+\.){3}0x[0-9a-f]+", host))


def registered_domain(host: str) -> str:
    if not host or _is_ip(host):
        return host
    labels = host.split(".")
    if len(labels) >= 3 and labels[-2] in SECOND_LEVEL and len(labels[-1]) == 2:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def _edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def is_official_domain(reg: str) -> bool:
    return reg in BRAND_DOMAINS or reg in LOCAL_DOMAINS


def _is_adjacent_swap(token: str, brand: str) -> bool:
    if len(token) != len(brand) or token == brand:
        return False
    diff = [i for i, (a, b) in enumerate(zip(token, brand)) if a != b]
    return len(diff) == 2 and diff[1] == diff[0] + 1 and token[diff[0]] == brand[diff[1]] \
        and token[diff[1]] == brand[diff[0]]


def _plain(text: str) -> str:
    if text.startswith("xn--"):
        try:
            text = text.encode("ascii").decode("idna")
        except UnicodeError:
            pass
    text = text.translate(CONFUSABLES)
    for fake, real in LOOKALIKES:
        text = text.replace(fake, real)
    return text


def lookalike_brand(reg_main: str):
    if not reg_main or reg_main in BRANDS or reg_main in LOCAL_BRANDS:
        return None
    whole = _plain(reg_main)
    for name, brand in LOCAL_DOMAIN_NAMES.items():
        if reg_main != name and (whole.replace("-", "") == name.replace("-", "")
                                 or _is_adjacent_swap(reg_main, name)):
            return brand
    for token in sorted({reg_main, *reg_main.split("-")}):
        plain = _plain(token)
        for brand in (*BRANDS, *LOCAL_BRANDS):
            if token == brand:
                continue
            if plain == brand or _is_adjacent_swap(token, brand):
                return brand
            allowed = 2 if len(brand) >= 9 else 1 if len(brand) >= 6 else 0
            if allowed and abs(len(plain) - len(brand)) <= allowed \
                    and _edit_distance(plain, brand) <= allowed:
                return brand
    return None


def local_brand_in(text: str, skip: str = ""):
    for brand in LOCAL_BRANDS:
        if brand == skip:
            continue
        if brand in SHORT_LOCAL_BRANDS:
            if re.search(rf"(?<![a-z0-9]){brand}(?![a-z0-9])", text):
                return brand
        elif brand in text:
            return brand
    return None


def _is_lookalike(reg_main: str) -> bool:
    return lookalike_brand(reg_main) is not None


def extract_features(url: str) -> dict:
    raw, host, path, has_scheme = _parse(str(url))
    lower = raw.lower()

    is_ip = _is_ip(host)
    reg = registered_domain(host)
    reg_main = reg.split(".")[0] if reg and not is_ip else ""
    tld = host.rsplit(".", 1)[-1] if "." in host and not is_ip else ""

    subdomain_part = host[: -len(reg)].rstrip(".") if reg and host.endswith(reg) else ""
    num_subdomains = 0 if is_ip or not subdomain_part else len(subdomain_part.split("."))

    official_local = reg in LOCAL_DOMAINS
    on_platform = any(host == h or host.endswith("." + h) for h in HOSTING_PLATFORMS)
    official_brand = reg_main in BRANDS and is_official_domain(reg)
    wrong_brand_domain = reg_main in BRANDS and not official_brand and not on_platform

    brand_not_in_domain = int(any(b in lower and b != reg_main for b in BRANDS)
                              or (not official_local and local_brand_in(lower, reg_main) is not None))

    path_segments = [p for p in path.split("/") if p]
    digits = sum(ch.isdigit() for ch in raw)

    features = {
        "url_length": len(raw),
        "domain_length": len(host),
        "num_dots": host.count("."),
        "num_subdomains": num_subdomains,
        "num_hyphens": host.count("-"),
        "has_ip": int(is_ip),
        "has_at": int("@" in raw),
        "is_https": int(lower.startswith("https://")),
        "num_suspicious_words": sum(lower.count(w) for w in SUSPICIOUS_WORDS),
        "brand_not_in_domain": brand_not_in_domain,
        "suspicious_tld": int(tld in SUSPICIOUS_TLDS),
        "digit_ratio": round(digits / len(raw), 4) if raw else 0.0,
        "entropy": round(shannon_entropy(lower), 4),
        "path_depth": len(path_segments),
        "is_shortener": int(reg in SHORTENERS or host in SHORTENERS),
        "brand_is_domain": int(official_brand or official_local),
        "suspicious_in_domain": sum(host.count(w) for w in SUSPICIOUS_WORDS),
        "brand_lookalike": int(wrong_brand_domain or (not official_local and _is_lookalike(reg_main))),
        "hosting_platform": int(on_platform),
    }
    return features


def features_dataframe(urls):
    import pandas as pd
    rows = [extract_features(u) for u in urls]
    return pd.DataFrame(rows, columns=FEATURE_NAMES)


def has_scheme(url: str) -> bool:
    return _parse(str(url))[3]


def url_text(url: str) -> str:
    return re.sub(r"^[a-z][a-z0-9+.-]*://", "", _parse(str(url))[0].lower())


def url_domain(url: str) -> str:
    return registered_domain(_parse(str(url))[1])


def url_key(url: str) -> str:
    return url_text(url).rstrip("/")


if __name__ == "__main__":
    for u in [
        "https://www.google.com",
        "http://paypal-secure.login-verify.xyz/update",
        "http://193.42.11.8/bank/verify/index.php",
        "https://bit.ly/3xYz12",
        "https://accounts.google.com.security-check.top/signin?id=88213",
    ]:
        print(u)
        for k, v in extract_features(u).items():
            print(f"   {FEATURE_LABELS[k]:<38} {v}")
        print()
