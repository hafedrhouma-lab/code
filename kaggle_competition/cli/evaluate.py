# cli/evaluate.py
from __future__ import annotations

import argparse

from model.infer.evaluate import run_evaluation


def _parse_args():
    ap = argparse.ArgumentParser(
        description="Evaluate a saved model on TRAIN/TEST for a given cutoff."
    )
    ap.add_argument("--model", required=True, help="Path to saved joblib (pipeline + threshold)")
    ap.add_argument("--raw", required=True, help="Path to raw JSON data")
    ap.add_argument("--cutoff", required=True, help="Cutoff date for TRAIN (<) and TEST (>=")
    ap.add_argument("--out_dir", default="artifacts/eval", help="Where to write metrics JSON")

    ap.add_argument("--threshold", type=float, default=None, help="Manual single threshold")
    ap.add_argument("--thresholds", type=str, default=None, help="Comma-separated thresholds")

    ap.add_argument(
        "--recompute_threshold", action="store_true", help="Recompute robust threshold via CV"
    )
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--recall_target", type=float, default=1.0)
    ap.add_argument("--thr_agg", choices=["median", "p25", "p10"], default="median")
    ap.add_argument("--thr_use_max_precision", action="store_true")
    return ap.parse_args()


def main():
    a = _parse_args()
    run_evaluation(
        model_path=a.model,
        raw_path=a.raw,
        cutoff=a.cutoff,
        out_dir=a.out_dir,
        manual_threshold=a.threshold,
        manual_thresholds=a.thresholds,
        recompute_threshold=a.recompute_threshold,
        n_splits=a.n_splits,
        recall_target=a.recall_target,
        thr_agg=a.thr_agg,
        thr_use_max_precision=a.thr_use_max_precision,
    )


if __name__ == "__main__":
    main()
