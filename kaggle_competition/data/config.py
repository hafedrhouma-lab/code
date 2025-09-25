from dataclasses import dataclass


@dataclass(frozen=True)
class Defaults:
    cutoff_date: str = "2019-01-01"


@dataclass(frozen=True)
class Paths:
    raw_json: str = "data/datasets/dataset.json"
    artifacts_dir: str = "artifacts/run1"
