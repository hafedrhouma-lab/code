import logging
import os
from pathlib import Path
from typing import Any, Optional

import cloudpickle
import mlflow
import newrelic.agent
import numpy as np
import structlog
from mlflow.entities.model_registry import ModelVersion, RegisteredModel

from base.v0.mlclass import MlflowBase
from base.v0.ml_metadata import find_model_by_metadata
from base.v0.perf import perf_manager
from projects.vendor_ranking.common.two_towers.src.cli import (
    validate_features_names, WARMUP_USER_MODEL_INPUT,
    GUEST_USER_FEATURES_NAMES, NAMES_MAPPING, BASE_USER_FEATURES_NAMES
)
from projects.vendor_ranking.common.two_towers.src.cli.predict import TwoTowersPredictor


LOG: "structlog.stdlib.BoundLogger" = structlog.getLogger(__name__)


class ModelWrapper(MlflowBase):
    def __init__(self) -> None:
        self.two_tower_model = None
        self.model_name = os.environ['MLFLOW_EXPERIMENT_NAME']
        self.user_model = None
        self.chain_model = None

        self.model_info: Optional[RegisteredModel] = None
        self.model_version: Optional[ModelVersion] = None

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        with perf_manager(
                description="Successfully loaded the model",
                description_before=f"-- Loading model context {context.artifacts} --"
        ):
            LOG.info("Loading parameter input data")
            with open(context.artifacts["params_input"], "rb") as params_input_data:
                params_input_data = cloudpickle.load(params_input_data)
            LOG.info("Successfully loaded parameter input data")

            user_weights_file = os.path.join(context.artifacts["user_weights"], "user_model_weights")
            chain_weights_file = os.path.join(context.artifacts["chain_weights"], "chain_model_weights")

            self.two_tower_model = TwoTowersPredictor(
                params_input_data,
                context.artifacts["model_config"],
                context.artifacts["ese_chain_embeddings"],
                user_weights_file,
                chain_weights_file
            )

        if new_relic_app_name := os.environ.get("NEW_RELIC_APP_NAME"):
            newrelic_config_file = Path(__file__).parent / "newrelic-agent.ini"
            assert newrelic_config_file.exists(), f"New Relic configuration file not found: {newrelic_config_file}"
            newrelic.agent.initialize(config_file=str(newrelic_config_file))
            newrelic.agent.register_application()

            LOG.info(
                f"New Relic agent initialized: "
                f"config file = {newrelic_config_file}, app name = {new_relic_app_name}"
            )
        else:
            LOG.warning(
                f"New Relic agent IS NOT initialized. Env var `NEW_RELIC_APP_NAME` not found"
            )

        self.model_info, self.model_version = find_model_by_metadata()
        if self.model_info and self.model_version:
            LOG.info(f"Model name: {self.model_info.name}, version: {self.model_version.version}")

        validate_features_names(WARMUP_USER_MODEL_INPUT.keys(), BASE_USER_FEATURES_NAMES)
        validate_features_names(WARMUP_USER_MODEL_INPUT.keys(), GUEST_USER_FEATURES_NAMES)

    @classmethod
    def validate_request(cls, request: dict[str, list[Any]]) -> dict[str, Any]:
        return {
            NAMES_MAPPING.get(key, key): value[0]  # get the first values
            for key, value in request.items()
        }

    @newrelic.agent.function_trace()
    def predict(
            self,
            context: mlflow.pyfunc.PythonModelContext,
            request: dict[str, list[Any]]
    ) -> dict[str, "np.ndarray"]:
        newrelic.agent.set_transaction_name("VR_2T_EG")
        model_input: dict[str, Any] = self.validate_request(request)

        embeddings = self.two_tower_model.get_embeddings(
            tower_name="user_tower",
            model_input=model_input
        )
        return {
            "value": embeddings,
            "version": self.model_version and self.model_version.version,
            "model": self.model_info and self.model_info.name
        }
