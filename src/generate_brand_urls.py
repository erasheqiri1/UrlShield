import random
from pathlib import Path

import pandas as pd

from src.features import LOCAL_DOMAINS, _is_lookalike

OUT = Path(__file__).resolve().parent.parent / "data" / "generated_brand_urls.csv"
LOCAL_OUT = OUT.with_name("generated_local_brand_urls.csv")

DOMAINS = {
    "paypal": "paypal.com", "apple": "apple.com", "icloud": "icloud.com",
    "microsoft": "microsoft.com", "outlook": "outlook.com", "google": "google.com",
    "gmail": "gmail.com", "facebook": "facebook.com", "instagram": "instagram.com",
    "whatsapp": "whatsapp.com", "amazon": "amazon.com", "netflix": "netflix.com",
    "ebay": "ebay.com", "dhl": "dhl.com", "fedex": "fedex.com", "binance": "binance.com",
    "coinbase": "coinbase.com", "linkedin": "linkedin.com", "dropbox": "dropbox.com",
    "adobe": "adobe.com", "github": "github.com", "youtube": "youtube.com",
    "twitter": "twitter.com", "yahoo": "yahoo.com", "telegram": "telegram.org",
    "tiktok": "tiktok.com", "spotify": "spotify.com", "discord": "discord.com",
}
PATHS = ["", "/login", "/signin", "/account", "/help", "/about", "/settings", "/support",
         "/security", "/search?q=test", "/news/today"]
SWAPS = [("l", "1"), ("o", "0"), ("e", "3"), ("s", "5"), ("m", "rn"), ("w", "vv"), ("i", "1")]


def typos(brand: str) -> list:
    out = {brand + "1", brand + "2", brand + "s", brand + brand[-1],
           brand + "-login", brand + "-secure"}
    for real, fake in SWAPS:
        if real in brand:
            out.add(brand.replace(real, fake, 1))
    for i in range(1, len(brand) - 1):
        out.add(brand[:i] + brand[i + 1:])
        out.add(brand[:i] + brand[i + 1] + brand[i] + brand[i + 2:])
    return sorted(t for t in out if t != brand and _is_lookalike(t))


LOCAL_PATHS = ["", "/login", "/hyrje", "/verifikim", "/llogaria", "/kontakt", "/rreth-nesh",
               "/sherbimet", "/ndihme"]
LOCAL_WORDS = ["login", "online", "secure", "verifikim", "llogaria", "ebanking", "kos", "ks"]


def local_imitations(brand: str) -> list:
    out = {f"{brand}-{word}" for word in LOCAL_WORDS} | {f"my-{brand}", f"{brand}-kos-login"}
    if len(brand) > 4:
        out |= set(typos(brand))
    else:
        out |= {brand + "1", brand + "s"}
    return sorted(out)


def generate_local(brands=None, seed: int = 3) -> pd.DataFrame:
    rng = random.Random(seed)
    brands = sorted(set(LOCAL_DOMAINS.values())) if brands is None else brands
    rows = []
    for brand in brands:
        for fake in local_imitations(brand):
            for tld in rng.sample(["com", "net", "org", "info", "xyz", "top"], 2):
                rows.append((f"https://{fake}.{tld}{rng.choice(LOCAL_PATHS)}", 0))
    for domain, brand in LOCAL_DOMAINS.items():
        if brand in brands:
            rows.extend((f"https://{domain}{path}", 1) for path in LOCAL_PATHS)
    return pd.DataFrame(rows, columns=["URL", "label"]).drop_duplicates("URL")


def generate(brands, seed: int = 1) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for brand in brands:
        for typo in typos(brand):
            for tld in rng.sample(["com", "net", "org"], 2):
                rows.append((f"https://{typo}.{tld}{rng.choice(PATHS)}", 0))
        rows.extend((f"https://{DOMAINS[brand]}{path}", 1) for path in PATHS)
    return pd.DataFrame(rows, columns=["URL", "label"]).drop_duplicates("URL")


if __name__ == "__main__":
    df = generate(sorted(DOMAINS))
    df.to_csv(OUT, index=False)
    print(f"U ruajt: {OUT} · {len(df)} URL · imitime: {int((df.label == 0).sum())} · "
          f"faqe të vërteta: {int((df.label == 1).sum())}")
    local = generate_local()
    local.to_csv(LOCAL_OUT, index=False)
    print(f"U ruajt: {LOCAL_OUT} · {len(local)} URL · imitime: {int((local.label == 0).sum())} · "
          f"faqe të vërteta: {int((local.label == 1).sum())}")
