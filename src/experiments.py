import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from src.features import FEATURE_NAMES, MODEL_FEATURES, TEXT_FEATURE, features_dataframe, url_domain, url_key, url_text
from src.train import (GENERATED_SOURCES, MODELS_DIR, ROOT, build_models, build_text_model, group_folds,
                       load_training_data, text_scores_out_of_fold)

MODELS = ["Random Forest", "XGBoost"]


def scores(y, pred) -> dict:
    return {"P": round(precision_score(y, pred, zero_division=0), 3), "R": round(recall_score(y, pred, zero_division=0), 3),
            "F1": round(f1_score(y, pred, zero_division=0), 3)}


def main():
    ap = argparse.ArgumentParser(description="Eksperimentet e kontributit")
    ap.add_argument("--data", required=True)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--test-size", type=float, default=0.2)
    args = ap.parse_args()

    meta = json.loads((MODELS_DIR / "metrics.json").read_text(encoding="utf-8"))
    df, _, _ = load_training_data(args.data, args.sample, args.extra, verbose=False)
    X = features_dataframe(df["url"].tolist())
    y = df["y"].to_numpy()
    texts = np.array([url_text(u) for u in df["url"]], dtype=object)
    groups = np.array([url_domain(u) for u in df["url"]], dtype=object)
    train_idx, test_idx = next(group_folds(round(1 / args.test_size)).split(X, y, groups))
    generated = df["source"].isin(GENERATED_SOURCES).to_numpy()

    exam = pd.read_csv(ROOT / "data" / "phreshphish_test.csv")
    exam = exam[~exam["url"].map(url_key).isin(set(df["url"].map(url_key)))]
    exam_y = (exam["label"] == "phishing").astype(int).to_numpy()
    exam_X = features_dataframe(exam["url"].tolist())
    exam_text = np.array([url_text(u) for u in exam["url"]], dtype=object)

    test_real = test_idx[~generated[test_idx]]
    test_fake = test_idx[generated[test_idx] & (y[test_idx] == 1)]
    variants = [
        ("Vetëm 19 veçoritë", FEATURE_NAMES, False, True),
        ("Vetëm modeli i shkronjave", None, False, True),
        ("19 veçoritë + shkronjat (modeli final)", MODEL_FEATURES, False, True),
        ("Modeli final pa URL sintetike në trajnim", MODEL_FEATURES, True, True),
    ]
    rows = []
    for label, columns, drop_generated, _ in variants:
        fit = train_idx[~generated[train_idx]] if drop_generated else train_idx
        text_model = build_text_model().fit(texts[fit], y[fit])
        tx = lambda idx: text_model.predict_proba(texts[idx])[:, 1]
        exam_score = text_model.predict_proba(exam_text)[:, 1]
        if columns is None:
            preds = {"Modeli i shkronjave": (tx(test_real) >= .5, tx(test_fake) >= .5, exam_score >= .5)}
        else:
            X_fit = X.iloc[fit].assign(**{TEXT_FEATURE: text_scores_out_of_fold(texts[fit], y[fit], groups[fit])})
            frame = lambda idx: X.iloc[idx].assign(**{TEXT_FEATURE: tx(idx)})[columns]
            preds = {}
            for name in MODELS:
                model = build_models()[name].set_params(**meta["models"][name]["params"]).fit(X_fit[columns], y[fit])
                preds[name] = (model.predict(frame(test_real)), model.predict(frame(test_fake)),
                               model.predict(exam_X.assign(**{TEXT_FEATURE: exam_score})[columns]))
        for name, (p_real, p_fake, p_exam) in preds.items():
            real, big = scores(y[test_real], p_real), scores(exam_y, p_exam)
            rows.append({"Eksperimenti": label, "Modeli": name,
                         "F1 hold-out real": real["F1"], "F1 PhreshPhish-test": big["F1"],
                         "P PhreshPhish-test": big["P"], "R PhreshPhish-test": big["R"],
                         "Imitime sintetike të kapura": round(float(np.mean(p_fake)), 3)})
            print(rows[-1], flush=True)
    result = {
        "rows": rows,
        "note": (f"Hold-out real: {len(test_real):,} URL pa ato sintetike. PhreshPhish-test: {len(exam):,} URL. "
                 f"Imitime sintetike: {len(test_fake):,} URL të krijuara nga gjeneratori, etiketa e tyre është supozim. "
                 "Të gjitha variantet trajnohen në të njëjtën ndarje sipas domenit."),
    }
    (MODELS_DIR / "experiments.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("U ruajt: models/experiments.json")


if __name__ == "__main__":
    main()
