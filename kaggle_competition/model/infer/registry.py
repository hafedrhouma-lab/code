# model/infer/registry.py
from __future__ import annotations

from collections.abc import Callable

# Each registered function must return a pandas DataFrame of importances.
ImportanceFn = Callable[..., "pd.DataFrame"]  # type: ignore[name-defined]

IMPORTANCE_REGISTRY: dict[str, ImportanceFn] = {}


def register_importance(name: str):
    def _wrap(fn: ImportanceFn) -> ImportanceFn:
        IMPORTANCE_REGISTRY[name] = fn
        return fn

    return _wrap
