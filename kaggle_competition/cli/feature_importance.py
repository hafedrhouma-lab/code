# cli/feature_importance.py
from __future__ import annotations

import argparse
import importlib

from model.infer.runner import run_feature_importance

importlib.import_module("model.infer.importance")


def _parse_args():
    ap = argparse.ArgumentParser(
        description="Dump (and optionally plot) feature importances from a saved model artifact."
    )
    ap.add_argument("--model", required=True, help="Path to saved joblib")
    ap.add_argument("--out_csv", default="artifacts/fi/feature_importance.csv")
    ap.add_argument("--topk", type=int, default=None)

    ap.add_argument("--method", choices=["built_in", "permutation", "shap"], default="built_in")
    ap.add_argument("--raw", default=None, help="Required for permutation / shap")
    ap.add_argument("--cutoff", default=None, help="Required for permutation / shap")

    ap.add_argument("--n_repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scoring", choices=["roc_auc", "average_precision"], default="roc_auc")
    ap.add_argument("--perm_rows", type=int, default=4000)
    ap.add_argument("--allow_dense_full", action="store_true")

    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--plot_path", default=None)
    return ap.parse_args()


def main():
    a = _parse_args()
    run_feature_importance(
        model_path=a.model,
        out_csv=a.out_csv,
        topk=a.topk,
        method=a.method,
        raw_path=a.raw,
        cutoff=a.cutoff,
        n_repeats=a.n_repeats,
        seed=a.seed,
        scoring=a.scoring,
        perm_rows=a.perm_rows,
        allow_dense_full=a.allow_dense_full,
        plot=a.plot,
        plot_path=a.plot_path,
    )


if __name__ == "__main__":
    main()
