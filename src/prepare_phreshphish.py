from pathlib import Path

import numpy as np
import pandas as pd

from src.features import HOSTING_PLATFORMS, url_domain, url_key
from src.train import group_folds, load_dataset

DATA = Path(__file__).resolve().parent.parent / "data"
PER_CLASS = 5000
PER_DOMAIN = 200
VALIDATION_CLASS_SIZE = 8000


def safe_domain(url: str) -> str:
    try:
        return url_domain(url)
    except ValueError:
        return ""


def base_data() -> pd.DataFrame:
    phi, _ = load_dataset(DATA / "PhiUSIIL_Phishing_URL_Dataset.csv", "URL", "label", "0")
    base = phi.sample(20000, random_state=42)
    for name in ["hannousse_phishing_urls.csv", "generated_brand_urls.csv", "generated_local_brand_urls.csv"]:
        extra, _ = load_dataset(DATA / name, "URL", "label", "0")
        base = pd.concat([base, extra[~extra["url"].isin(base["url"])]], ignore_index=True)
    base = base.reset_index(drop=True)
    base["dom"] = base["url"].map(url_domain)
    return base


def to_training_format(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"URL": df["url"], "label": 1 - df["y"]})


def main():
    base = base_data()
    _, holdout_idx = next(group_folds(5).split(base, base["y"], base["dom"]))
    platforms = set(HOSTING_PLATFORMS)
    holdout_domains = set(base["dom"].iloc[holdout_idx]) - platforms

    pp = pd.read_csv(DATA / "phreshphish_urls.csv").dropna()
    pp["url"] = pp["url"].astype(str).str.strip()
    pp["dom"] = pp["url"].map(safe_domain)
    pp = pp[pp["dom"] != ""]
    pp["key"] = pp["url"].map(url_key)
    pp = pp.drop_duplicates("key")
    pp["y"] = (pp["label"] == "phish").astype(int)
    pp = pp[~pp["key"].isin(set(base["url"].map(url_key)))]

    test_all = pp[pp["split"] == "test"]
    test_domains = set(test_all["dom"]) - platforms
    pool = pp[(pp["split"] == "train") & ~pp["dom"].isin(holdout_domains | test_domains)]
    pool = pool.sample(frac=1, random_state=42).groupby("dom", sort=False).head(PER_DOMAIN).reset_index(drop=True)

    domains = np.array(sorted(pool["dom"].unique()), dtype=object)
    np.random.default_rng(42).shuffle(domains)
    candidates = pool[~pool["dom"].isin(set(domains[: len(domains) // 5]))]
    train = pd.concat([g.sample(min(PER_CLASS, len(g)), random_state=2) for _, g in candidates.groupby("y")])

    seen_domains = (set(base["dom"]) | set(train["dom"])) - platforms
    exam = test_all[~test_all["dom"].isin(seen_domains)]
    exam = pd.concat([g.sample(min(15000, len(g)), random_state=3) for _, g in exam.groupby("y")])

    to_training_format(train).to_csv(DATA / "phreshphish_train.csv", index=False, encoding="utf-8")
    pd.DataFrame({"url": exam["url"], "label": np.where(exam["y"] == 1, "phishing", "legjitime")}).to_csv(
        DATA / "phreshphish_test.csv", index=False, encoding="utf-8")
    print(f"Trajnim: {len(train)} URL · {train['y'].value_counts().to_dict()}")
    print(f"Provim: {len(exam)} URL · {exam['y'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
