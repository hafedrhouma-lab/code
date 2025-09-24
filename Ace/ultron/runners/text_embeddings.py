#!/usr/bin/env python3

import os
from enum import Enum
from typing import Optional

import bentoml
from sentence_transformers import SentenceTransformer

from ultron import SERVICE_NAME

MODEL_TAG = f"{SERVICE_NAME}:text-embeddings"

_runner: Optional[bentoml.Runner] = None


class TextEmbeddingsModelName(str, Enum):
    ALL_MINILM_L6_V2 = "all-MiniLM-L6-v2"


def get_runner() -> bentoml.Runner:
    global _runner
    if not _runner:
        _runner = bentoml.pytorch.get(MODEL_TAG).to_runner()

    return _runner


def reload() -> bentoml.Model:
    """
    1) Delete current model.
    2) Download model and store in `bemtoml` compatible format
    """
    try:
        bentoml.models.delete(MODEL_TAG)
    except bentoml.exceptions.NotFound:
        pass

    model = SentenceTransformer(TextEmbeddingsModelName.ALL_MINILM_L6_V2.value)
    bemtoml_model = bentoml.pytorch.save_model(
        name=MODEL_TAG,
        model=model,
        signatures={"encode": {"batch_dim": 0, "batchable": True}},
    )
    return bemtoml_model


if __name__ == "__main__":
    loaded_model = reload()
    loaded_model.export(os.path.dirname(__file__) + "/../../models")
    print(f"Bento model tag: {loaded_model.tag}")
