#!/usr/bin/env python3

import typing as t
from contextlib import suppress
from pathlib import Path
from typing import Optional, Tuple
from unittest import mock

import bentoml
from sentence_transformers import CrossEncoder

from ultron import SERVICE_NAME

MODEL_TAG: str = f"{SERVICE_NAME}:cross-encoder-model"
TOKENIZER_TAG: str = f"{SERVICE_NAME}:cross-encoder-tokenizer"

MODEL_FILES_DIR: Path = (Path(__file__).parent / "../../models").resolve()
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_runner: Optional[bentoml.Runner] = None


class CrossEncoderRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("cpu",)
    SUPPORTS_CPU_MULTI_THREADING = False

    def __init__(self, model_tag: str, tokenizer_tag: str):
        model = bentoml.transformers.load_model(bentoml.transformers.get(model_tag))
        with mock.patch(
            "transformers.AutoModelForSequenceClassification.from_pretrained", new=lambda *args, **kwargs: model
        ):
            tokenizer = bentoml.transformers.load_model(bentoml.transformers.get(tokenizer_tag))
            with mock.patch("transformers.AutoTokenizer.from_pretrained", new=lambda *args, **kwargs: tokenizer):
                self.model = CrossEncoder(CROSS_ENCODER_MODEL_NAME)

    @bentoml.Runnable.method(batchable=False)
    def predict(self, request: list[Tuple[str, str]]) -> list[float]:
        return self.model.predict(request)


def get_runner() -> bentoml.Runner:
    global _runner
    if not _runner:
        _runner = t.cast(
            "RunnerImpl",  # noqa: F821
            bentoml.Runner(
                CrossEncoderRunnable,
                name="cross_encoder",
                runnable_init_params={
                    "model_tag": MODEL_TAG,
                    "tokenizer_tag": TOKENIZER_TAG,
                },
                models=[bentoml.transformers.get(tag) for tag in (MODEL_TAG, TOKENIZER_TAG)],
            ),
        )

    return _runner


def reload() -> tuple[bentoml.Model, bentoml.Model]:
    """
    1) Delete current model.
    2) Download model and store in `bemtoml` compatible format
    """
    for tag in (MODEL_TAG, TOKENIZER_TAG):
        with suppress(bentoml.exceptions.NotFound):
            bentoml.models.delete(tag)

    encoder = CrossEncoder(CROSS_ENCODER_MODEL_NAME)
    bemtoml_model = bentoml.transformers.save_model(MODEL_TAG, encoder.model)
    bemtoml_tokenizer = bentoml.transformers.save_model(TOKENIZER_TAG, encoder.tokenizer)
    return bemtoml_model, bemtoml_tokenizer


if __name__ == "__main__":
    loaded_model, loaded_tokenizer = reload()
    model_files_dir = str(MODEL_FILES_DIR)
    loaded_model.export(model_files_dir)
    loaded_tokenizer.export(model_files_dir)
    print(f"Bento model tag: {loaded_model.tag}, tokenizer tag: {loaded_tokenizer.tag}")
