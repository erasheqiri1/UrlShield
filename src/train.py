import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import joblib
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import ParameterGrid, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.features import MODEL_FEATURES, TEXT_FEATURE, features_dataframe, url_domain, url_key, url_text

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
RANDOM_STATE = 42
CV_FOLDS = 5
GENERATED_SOURCES = {"generated_brand_urls.csv", "generated_local_brand_urls.csv"}

PARAM_GRIDS = {
    "Logistic Regression": {"logisticregression__C": [0.1, 1, 10]},
    "Decision Tree": {"max_depth": [8, 12, 16], "min_samples_leaf": [1, 5, 20]},
    "Random Forest": {"max_depth": [None, 20], "min_samples_leaf": [1, 2, 5]},
    "Gradient Boosting": {"n_estimators": [200, 400], "max_depth": [3, 4],
                          "learning_rate": [0.05, 0.1]},
    "XGBoost": {"n_estimators": [300, 500], "max_depth": [4, 6, 8],
                "learning_rate": [0.05, 0.1]},
}


def build_models():
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=12, min_samples_leaf=5, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_STATE
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=3, random_state=RANDOM_STATE
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
            colsample_bytree=0.8, eval_metric="logloss", n_jobs=-1, random_state=RANDOM_STATE
        ),
    }


def build_text_model():
    return make_pipeline(
        TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=20000,
                        sublinear_tf=True, min_df=3),
        LogisticRegression(C=10, max_iter=2000, random_state=RANDOM_STATE),
    )


def group_folds(n_splits: int, seed: int = RANDOM_STATE):
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def text_scores_out_of_fold(texts, y, groups, n_splits: int = 4):
    scores = np.zeros(len(y))
    for fit_idx, score_idx in group_folds(n_splits).split(texts, y, groups):
        model = build_text_model().fit(texts[fit_idx], y[fit_idx])
        scores[score_idx] = model.predict_proba(texts[score_idx])[:, 1]
    return scores


def slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def compute_metrics(y_true, y_pred) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def roc_points(y_true, proba, n_points: int = 100) -> dict:
    fpr, tpr, _ = roc_curve(y_true, proba)
    keep = np.unique(np.linspace(0, len(fpr) - 1, n_points).astype(int))
    return {"fpr": [round(float(v), 4) for v in fpr[keep]],
            "tpr": [round(float(v), 4) for v in tpr[keep]]}


def feature_importance(name, model) -> dict:
    if name == "Logistic Regression":
        coefs = model[-1].coef_[0]
        values = np.abs(coefs)
    else:
        values = model.feature_importances_
    total = values.sum() or 1
    return {f: round(float(v / total), 4) for f, v in zip(MODEL_FEATURES, values)}


def cross_validate(models, X, y, texts, groups, n_splits: int = CV_FOLDS) -> dict:
    per_model = {name: {"f1": [], "roc_auc": []} for name in models}
    for i, (fit_idx, val_idx) in enumerate(group_folds(n_splits).split(X, y, groups), 1):
        X_fit = X.iloc[fit_idx].assign(
            **{TEXT_FEATURE: text_scores_out_of_fold(texts[fit_idx], y[fit_idx], groups[fit_idx], 3)})
        text_model = build_text_model().fit(texts[fit_idx], y[fit_idx])
        X_val = X.iloc[val_idx].assign(**{TEXT_FEATURE: text_model.predict_proba(texts[val_idx])[:, 1]})
        for name, params in models.items():
            model = build_models()[name].set_params(**params).fit(X_fit[MODEL_FEATURES], y[fit_idx])
            proba = model.predict_proba(X_val[MODEL_FEATURES])[:, 1]
            per_model[name]["f1"].append(f1_score(y[val_idx], proba >= 0.5))
            per_model[name]["roc_auc"].append(roc_auc_score(y[val_idx], proba))
        print(f"   ndarja {i}/{n_splits} u krye")
    return {
        name: {f"{metric}_{stat}": round(float(fn(values)), 4)
               for metric, values in scores.items() for stat, fn in (("mean", np.mean), ("std", np.std))}
        for name, scores in per_model.items()
    }


def text_folds(X, y, texts, groups, n_splits: int):
    prepared = []
    for fit_idx, val_idx in group_folds(n_splits).split(X, y, groups):
        X_fit = X.iloc[fit_idx].assign(
            **{TEXT_FEATURE: text_scores_out_of_fold(texts[fit_idx], y[fit_idx], groups[fit_idx], 3)})
        text_model = build_text_model().fit(texts[fit_idx], y[fit_idx])
        X_val = X.iloc[val_idx].assign(**{TEXT_FEATURE: text_model.predict_proba(texts[val_idx])[:, 1]})
        prepared.append((X_fit[MODEL_FEATURES], y[fit_idx], X_val[MODEL_FEATURES], y[val_idx]))
    return prepared


def _fold_f1(name, params, X_fit, y_fit, X_val, y_val):
    model = build_models()[name].set_params(**params).fit(X_fit, y_fit)
    return f1_score(y_val, model.predict(X_val))


def grid_search(X, y, texts, groups, n_splits: int = 3) -> dict:
    folds = text_folds(X, y, texts, groups, n_splits)
    best = {}
    for name in build_models():
        combos = list(ParameterGrid(PARAM_GRIDS[name]))
        scores = Parallel(n_jobs=-1)(delayed(_fold_f1)(name, p, *fold) for p in combos for fold in folds)
        means = np.array(scores).reshape(len(combos), len(folds)).mean(axis=1)
        best[name] = combos[int(means.argmax())]
        print(f"   {name}: {best[name]} · F1 {means.max():.4f}")
    return best


def load_dataset(path, url_col, label_col, phishing_label):
    df = pd.read_csv(path, usecols=[url_col, label_col])
    df = df.rename(columns={url_col: "url", label_col: "label_raw"})
    df = df.dropna(subset=["url", "label_raw"])
    before = len(df)
    df = df.drop_duplicates(subset="url")
    df["y"] = (df["label_raw"].astype(str) == str(phishing_label)).astype(int)
    return df, before - len(df)


def load_training_data(data, sample, extras, url_col="URL", label_col="label", phishing_label="0", verbose=True):
    df, n_dupes = load_dataset(data, url_col, label_col, phishing_label)
    if sample and sample < len(df):
        df = df.sample(sample, random_state=RANDOM_STATE)
    df["source"] = Path(data).name
    if verbose:
        print(f"   {len(df)} URL (u hoqën {n_dupes} dublikata) · "
              f"phishing: {int(df.y.sum())} · legjitime: {int((1 - df.y).sum())}")
    n_extra = 0
    for path in extras:
        extra, _ = load_dataset(path, url_col, label_col, phishing_label)
        extra = extra[~extra["url"].isin(df["url"])]
        extra = extra.assign(source=Path(path).name)
        n_extra += len(extra)
        df = pd.concat([df, extra], ignore_index=True)
        if verbose:
            print(f"   + {len(extra)} URL nga {Path(path).name} · gjithsej {len(df)} · "
                  f"phishing: {int(df.y.sum())} · legjitime: {int((1 - df.y).sum())}")
    return df.reset_index(drop=True), n_dupes, n_extra


def main():
    ap = argparse.ArgumentParser(description="Trajno modelet e URLShield")
    ap.add_argument("--data", required=True, help="Rruga te CSV-ja e dataset-it")
    ap.add_argument("--url-col", default="URL")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--phishing-label", default="0",
                    help="Vlera që tregon phishing në kolonën label (PhiUSIIL: 0)")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--sample", type=int, default=0,
                    help="Përdor vetëm N rreshta (për provë të shpejtë)")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="Një ose më shumë CSV shtesë me të njëjtat kolona dhe etiketa. Shtohen "
                         "të plota, edhe me --sample. Shërbejnë që modeli të shohë URL legjitime "
                         "me path dhe imitime të markave, që PhiUSIIL pothuajse nuk i ka.")
    ap.add_argument("--tune", action="store_true",
                    help="Kërko parametrat më të mirë me GridSearch (vetëm në pjesën e trajnimit)")
    args = ap.parse_args()

    MODELS_DIR.mkdir(exist_ok=True)

    print("1) Leximi i dataset-it…")
    df, n_dupes, n_extra = load_training_data(args.data, args.sample, args.extra, args.url_col,
                                              args.label_col, args.phishing_label)

    print("2) Nxjerrja e veçorive…")
    t0 = time.time()
    X = features_dataframe(df["url"].tolist())
    y = df["y"].to_numpy()
    texts = np.array([url_text(u) for u in df["url"]], dtype=object)
    groups = np.array([url_domain(u) for u in df["url"]], dtype=object)
    print(f"   {X.shape[1]} veçori për {X.shape[0]} URL ({time.time() - t0:.1f}s)")

    print("3) Ndarja train/test sipas domenit…")
    n_splits = round(1 / args.test_size)
    train_idx, test_idx = next(group_folds(n_splits).split(X, y, groups))
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_test, y_test = X.iloc[test_idx], y[test_idx]
    url_test = df["url"].to_numpy()[test_idx]
    source_test = df["source"].to_numpy()[test_idx]
    print(f"   trajnim: {len(X_train)} · testim: {len(X_test)} · "
          f"domene të përbashkëta: {len(set(groups[train_idx]) & set(groups[test_idx]))}")

    print("4) Modeli i shkronjave…")
    text_model = build_text_model().fit(texts[train_idx], y_train)
    joblib.dump(text_model, MODELS_DIR / "text_model.pkl")
    X_train = X_train.assign(**{TEXT_FEATURE: text_scores_out_of_fold(
        texts[train_idx], y_train, groups[train_idx])})
    X_test = X_test.assign(**{TEXT_FEATURE: text_model.predict_proba(texts[test_idx])[:, 1]})

    params = {name: {} for name in build_models()}
    if args.tune:
        print("5) GridSearch në pjesën e trajnimit…")
        params = grid_search(X.iloc[train_idx], y_train, texts[train_idx], groups[train_idx])

    print(f"6) Cross-validation me {CV_FOLDS} ndarje…")
    cv = cross_validate(params, X.iloc[train_idx], y_train, texts[train_idx], groups[train_idx])

    results = {}
    for name, model in build_models().items():
        print(f"7) Trajnimi: {name}…")
        model.set_params(**params[name])
        t0 = time.time()
        model.fit(X_train[MODEL_FEATURES], y_train)
        train_time = time.time() - t0
        y_pred = model.predict(X_test[MODEL_FEATURES])
        proba = model.predict_proba(X_test[MODEL_FEATURES])[:, 1]
        m = compute_metrics(y_test, y_pred)
        m["roc_auc"] = round(float(roc_auc_score(y_test, proba)), 4)
        m["roc_curve"] = roc_points(y_test, proba)
        m["cv"] = cv[name]
        m["params"] = {k: v for k, v in params[name].items()}
        m["train_time_s"] = round(train_time, 2)
        m["feature_importance"] = feature_importance(name, model)
        real = ~np.isin(source_test, list(GENERATED_SOURCES))
        m["real_only"] = dict(compute_metrics(y_test[real], y_pred[real]), n=int(real.sum()),
                              roc_auc=round(float(roc_auc_score(y_test[real], proba[real])), 4))
        m["by_source"] = {
            source: dict(compute_metrics(y_test[source_test == source], y_pred[source_test == source]),
                         n=int((source_test == source).sum()))
            for source in pd.unique(source_test)
        }

        wrong = np.where(y_pred != y_test)[0][:25]
        m["errors"] = [
            {"url": str(url_test[i]),
             "true": "phishing" if y_test[i] else "legjitime",
             "pred": "phishing" if y_pred[i] else "legjitime"}
            for i in wrong
        ]
        results[name] = m
        joblib.dump(model, MODELS_DIR / f"{slug(name)}.pkl")
        print(f"   Acc {m['accuracy']:.4f} · P {m['precision']:.4f} · R {m['recall']:.4f} · "
              f"F1 {m['f1']:.4f} · AUC {m['roc_auc']:.4f} · CV F1 {cv[name]['f1_mean']:.4f} · "
              f"{train_time:.1f}s")

    best = max(results, key=lambda k: results[k]["cv"]["f1_mean"])
    joblib.dump(joblib.load(MODELS_DIR / f"{slug(best)}.pkl"), MODELS_DIR / "best_model.pkl")
    joblib.dump({"urls": {url_key(u) for u in df["url"]}, "domains": set(groups)},
                MODELS_DIR / "train_urls.pkl")

    legit_reference = X_train.loc[y_train == 0, MODEL_FEATURES].median().round(4).to_dict()

    meta = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(Path(args.data).name),
        "extra_datasets": [Path(path).name for path in args.extra],
        "n_extra": int(n_extra),
        "n_total": int(len(df)),
        "n_duplicates_removed": int(n_dupes),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_domains": int(len(set(groups))),
        "n_phishing": int(df.y.sum()),
        "n_legit": int((1 - df.y).sum()),
        "test_size": args.test_size,
        "generated_sources": sorted(GENERATED_SOURCES),
        "split": "domain",
        "cv_folds": CV_FOLDS,
        "tuned": bool(args.tune),
        "sample": args.sample,
        "features": MODEL_FEATURES,
        "best_model": best,
        "legit_reference": legit_reference,
        "models": results,
    }
    (MODELS_DIR / "metrics.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                              encoding="utf-8")
    print(f"\nModeli më i mirë (sipas CV F1): {best}")
    print(f"U ruajt: {MODELS_DIR / 'best_model.pkl'} dhe metrics.json")


if __name__ == "__main__":
    main()
