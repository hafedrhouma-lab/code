# model/cv.py
from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np


class ForwardTimeSplit:
    """
    Forward-chaining (expanding window) splitter for time-ordered data.
    - Sorts indices by time (you pass TRAIN dates).
    - Splits into (n_splits + 1) contiguous chunks: C0 .. Cn
    - Fold i: train = concat(C0..Ci), valid = C(i+1)
    This preserves causality for model selection / HP search.
    """

    def __init__(self, dates: Iterable, n_splits: int = 5, min_train_size: int = 1):
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        self.n_splits = n_splits
        self.min_train_size = min_train_size

        dates = np.asarray(dates)
        # sort by datetime if present, fallback is still argsort
        self.order = np.argsort(dates)

    def split(self, X=None, y=None, groups=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        chunks = np.array_split(self.order, self.n_splits + 1)
        for i in range(self.n_splits):
            val_idx = chunks[i + 1]
            train_idx = np.concatenate(chunks[: i + 1]) if i >= 0 else np.array([], dtype=int)
            if train_idx.size < self.min_train_size or val_idx.size == 0:
                continue
            yield train_idx, val_idx

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits
