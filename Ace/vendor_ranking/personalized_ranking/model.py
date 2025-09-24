#!/usr/bin/env python3

import os
from typing import Optional

import bentoml
import catboost as cb

import ace.bentoml.catboost
from vendor_ranking import SERVICE_NAME

MODEL_TAG = f"{SERVICE_NAME}:personalized-v1_20231031"

FILE = "cat_v3_2023-10-18_2023-10-31"

_runner: Optional[bentoml.Runner] = None


def get_runner() -> bentoml.Runner:
    global _runner

    if not _runner:
        # _runner = bentoml.catboost.get(MODEL_TAG).to_runner()
        bentoml_model = bentoml.catboost.get(MODEL_TAG)
        _runner = bentoml.Runner(
            ace.bentoml.catboost.get_runnable(bentoml_model),
            name=str(bentoml_model.tag).replace(":", "."),
            models=[bentoml_model],
        )

    return _runner


def reload(path) -> bentoml.Model:
    try:
        bentoml.models.delete(MODEL_TAG)
    except bentoml.exceptions.NotFound:
        pass

    model = cb.CatBoostRanker()
    model.load_model(path)
    bemtoml_model = bentoml.catboost.save_model(MODEL_TAG, model, signatures={"predict": {"batchable": False}})

    return bemtoml_model


if __name__ == "__main__":
    loaded_model = reload(os.path.dirname(__file__) + "/" + FILE)
    loaded_model.export(os.path.dirname(__file__) + "/../../models")
    print(f"Bento model tag: {loaded_model.tag}")
