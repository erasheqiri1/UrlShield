import argparse
from pathlib import Path

import pandas as pd

from src.core import evaluate_on, save_fresh_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    args = ap.parse_args()

    result = evaluate_on(pd.read_csv(args.data))
    print(f"{result['n_total']} URL të reja · phishing: {result['n_phishing']} · "
          f"legjitime: {result['n_legit']} · u hoqën: {result['n_empty']} bosh, "
          f"{result['n_unknown_labels']} me label të panjohur, {result['n_duplicates']} të përsëritura, "
          f"{result['n_seen_removed']} të para në trajnim")
    for name, m in result["models"].items():
        print(f"{name:<20} Acc {m['accuracy']:.4f} · P {m['precision']:.4f} · "
              f"R {m['recall']:.4f} · F1 {m['f1']:.4f}")
    save_fresh_metrics(result, Path(args.data).name)
    print("U ruajt: models/fresh_metrics.json")


if __name__ == "__main__":
    main()
