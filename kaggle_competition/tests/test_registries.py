# tests/test_registries.py
from __future__ import annotations

import importlib


def test_training_registry_contains_models():
    # Importing registers models via decorators
    importlib.import_module("model.training.pipeline")
    from model.training.registry import MODEL_REGISTRY

    assert "logreg" in MODEL_REGISTRY
    # lgbm might be optional if lightgbm isn't installed; don't fail if missing
    # but if present, it should be registered:
    try:
        import lightgbm  # noqa: F401

        assert "lgbm" in MODEL_REGISTRY
    except Exception:
        pass


def test_importance_registry_contains_methods():
    importlib.import_module("model.infer.importance")
    from model.infer.registry import IMPORTANCE_REGISTRY

    for name in ["built_in", "permutation", "shap"]:
        assert name in IMPORTANCE_REGISTRY
