from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, TypeVar, Generic

import newrelic.agent
import numpy as np

from abstract_ranking.input import AbstractRankingRequest
from ace.newrelic import add_transaction_attrs, add_transaction_attr

if TYPE_CHECKING:
    from ace.model_log import LogEntry
    import tensorflow as tf
    import pandas as pd


T = TypeVar("T", bound=AbstractRankingRequest)


@newrelic.agent.function_trace()
def infer_user_embedding_two_tower(
    features: "pd.DataFrame", user_model: "tf.keras.Sequential"
):
    features = {
        feature_col: np.asarray(features[feature_col])
        for feature_col in features.columns
    }
    out = features
    for layer in user_model.layers:
        inp = out
        out = layer(inp)
    return out.numpy()[0]


class BaseLogic(Generic[T], ABC):
    NAME: ClassVar[str] = None
    SERVICE_NAME: ClassVar[str] = None
    MODEL_TAG: ClassVar[str] = None

    # Human-readable name of logic.
    # That name can be used to refer to logic encapsulated inside the class.
    NICKNAME: ClassVar[str | None] = None

    def __init__(self, request: T, exec_log: "LogEntry", **kwargs):
        self.request = request
        self.stats: list = []
        self.exec_log: "LogEntry" = exec_log
        exec_log.service_name = self.NAME
        assert self.NAME is not None, "logic name must be provided"
        assert self.SERVICE_NAME is not None, "service name must be provided"
        assert self.MODEL_TAG is not None, "model tag name must be provided"
        exec_log.model_tag = self.MODEL_TAG

    @abstractmethod
    async def prepare_features(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    async def sort(self) -> list[int]:
        raise NotImplementedError()

    @classmethod
    async def check_execution_requirements(cls, msg: str):
        """Check that everything required for logic's calculation is provided.
        Each logic can have its own unique requirements.
        Some requirements can be one of the next:
        - Required tables are in the DB.
        - Tables contain required data (for example, `snapshot_date` is fresh enough).
        - S3 files are present.
        - Content of the S3 files is valid.
        """
        pass

    def push_transaction_stats(self):
        if not self.stats:
            add_transaction_attr("failed_stats_push", 1, service_name=self.SERVICE_NAME)
        else:
            add_transaction_attrs(tuple(self.stats), service_name=self.SERVICE_NAME)
