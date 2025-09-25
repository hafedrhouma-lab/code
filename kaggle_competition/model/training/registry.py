# model/training/registry.py
from __future__ import annotations

from collections.abc import Callable
from typing import Any

BuildSpaceFn = Callable[[Any, float, int], dict]
MODEL_REGISTRY: dict[str, BuildSpaceFn] = {}


def register_model(name: str):
    def _wrap(fn: BuildSpaceFn) -> BuildSpaceFn:
        MODEL_REGISTRY[name] = fn
        return fn

    return _wrap
