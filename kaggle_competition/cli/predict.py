# cli/predict.py
from __future__ import annotations

import argparse

from model.infer.predict import run_predict


def _parse_args():
    ap = argparse.ArgumentParser(description="Score new data with saved model artifact.")
    ap.add_argument("--model", required=True, help="Path to saved joblib")
    ap.add_argument("--raw", required=True, help="Path to raw JSON")
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument(
        "--threshold", type=float, default=None, help="Manual threshold (overrides stored)"
    )
    return ap.parse_args()


def main():
    a = _parse_args()
    run_predict(
        model_path=a.model,
        raw_path=a.raw,
        out_csv=a.out,
        manual_threshold=a.threshold,
    )


if __name__ == "__main__":
    main()
