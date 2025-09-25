# tests/test_evaluate_math.py
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from model.evaluate import cv_thresholds_forward, subset_rows, threshold_for_recall


def test_threshold_for_recall_simple():
    y = np.array([0, 1, 1, 0, 1])
    p = np.array([0.1, 0.9, 0.8, 0.2, 0.7])
    thr = threshold_for_recall(y, p, target=2 / 3)  # need 2 out of 3 positives
    assert 0.7 <= thr <= 0.8


def test_subset_rows_various_types():
    import pandas as pd
    import scipy.sparse as sp

    idx = np.array([1, 3, 4])

    X_np = np.arange(20).reshape(5, 4)
    X_df = pd.DataFrame(X_np, columns=list("abcd"))
    X_sp = sp.csr_matrix(X_np)

    assert subset_rows(X_np, idx).shape == (3, 4)
    assert subset_rows(X_df, idx).shape == (3, 4)
    assert subset_rows(X_sp, idx).shape == (3, 4)


def test_cv_thresholds_forward_runs():
    rng = np.random.RandomState(0)
    X = rng.randn(100, 5)
    y = (rng.rand(100) < 0.3).astype(int)

    # simple chronological "dates" just as increasing numbers
    dates = np.arange(100)
    # forward splits: (train up to t, validate next chunk)
    from model.cv import ForwardTimeSplit

    splitter = ForwardTimeSplit(dates=dates, n_splits=3, min_train_size=10)
    splits = list(splitter.split())

    est = LogisticRegression(max_iter=200)
    thrs = cv_thresholds_forward(
        base_estimator=est,
        X=X,
        y=y,
        splits=splits,
        recall_target=0.6,
        use_max_precision=True,
    )
    assert len(thrs) == len(splits)
