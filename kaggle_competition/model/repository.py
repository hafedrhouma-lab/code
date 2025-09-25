# model/repository.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import joblib


@dataclass
class Artifact:
    pipeline: Any
    model: Any  # pipeline.named_steps['clf']
    fe: Any  # pipeline.named_steps['fe']
    threshold: float | None


class ArtifactRepository:
    """Persistence boundary: swap Joblib for cloud/DB later without touching callers."""

    def save(self, artifact_path: str, pipeline, threshold: float | None = None) -> None:
        joblib.dump({"pipeline": pipeline, "threshold": threshold}, artifact_path)

    def load(self, artifact_path: str) -> Artifact:
        obj = joblib.load(artifact_path)
        if not isinstance(obj, dict) or "pipeline" not in obj:
            raise ValueError("Artifact must be a dict with 'pipeline' and optional 'threshold'.")
        pipe = obj["pipeline"]
        thr = obj.get("threshold", None)
        steps = getattr(pipe, "named_steps", {})
        fe = steps.get("fe")
        clf = steps.get("clf")
        if fe is None or clf is None:
            raise ValueError("Pipeline must contain steps named 'fe' and 'clf'.")
        return Artifact(pipeline=pipe, model=clf, fe=fe, threshold=thr)
