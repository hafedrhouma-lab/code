# cli/hp_search.py
import argparse

from model.training.hpsearch import run as run_hpsearch


def _parse_args():
    ap = argparse.ArgumentParser(description="Time-aware HP search (Pipeline + robust threshold).")
    ap.add_argument("--raw", default="data/datasets/dataset.json")
    ap.add_argument("--artifacts", default="artifacts/hpsearch_run_final2")
    ap.add_argument("--cutoff", default="2019-01-01")
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--n_iter", type=int, default=30)
    ap.add_argument("--recall_target", type=float, default=1.0)
    ap.add_argument("--thr_agg", choices=["median", "p25", "p10"], default="median")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_tfidf", type=int, default=5000)
    ap.add_argument("--top_sig", type=int, default=300)
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--thr_use_max_precision", action="store_true")
    return ap.parse_args()


def main():
    a = _parse_args()
    run_hpsearch(
        raw_path=a.raw,
        artifacts=a.artifacts,
        cutoff=a.cutoff,
        n_splits=a.n_splits,
        n_iter=a.n_iter,
        recall_target=a.recall_target,
        thr_agg=a.thr_agg,
        seed=a.seed,
        max_tfidf=a.max_tfidf,
        top_sig=a.top_sig,
        n_jobs=a.n_jobs,
        thr_use_max_precision=a.thr_use_max_precision,
    )


if __name__ == "__main__":
    main()
