import datetime as dt
import os
import pickle
from functools import cached_property
from pathlib import Path
from typing import Optional, Any

import newrelic.agent
import structlog
import yaml
from tenacity import RetryCallState, retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from ace.configs.config import AppS3Config
from ace.storage.s3 import S3DownloadManager
from nba.input import VariantName

logger = structlog.get_logger()


def get_inference_path():
    base_path = os.path.join(os.getcwd(), "inference_model_artifacts")
    if not os.path.exists(base_path):
        os.makedirs(base_path, exist_ok=True)
    return base_path


def fallback_model_logic(retry_state: RetryCallState):
    # `attempt_number` starts with 1
    retry_state.kwargs["date_offset"] = retry_state.attempt_number


class AgentsManager:
    def __init__(
        self,
        s3_app_config: AppS3Config,
        base_dir: str,
        version: int = 2,
    ):
        self.s3_app_config: AppS3Config = s3_app_config
        self._bucket_name: str = s3_app_config.bucket_name

        self.base_local_dir: Path = Path(base_dir)
        self.model_config_file = "config/config.yaml"

        self.pickle_file_neural_ucb = "agents/UAE_NeuralUCB_reward/{today_no_dash}/UAE_NeuralUCB_reward.pkl"
        self.pickle_file_lin_ucb = "agents/UAE_LinUCB_reward/{today_no_dash}/UAE_LinUCB_reward.pkl"
        self.pickle_file_lin_ucb_i = "agents/UAE_LinUCB_iReward/{today_no_dash}/UAE_LinUCB_iReward.pkl"

        # neural regressor
        self.pickle_file_neural_regressor_i = (
            "agents/UAE_NeuralRegressor_iReward/{today_no_dash}/UAE_NeuralRegressor_iReward.pkl"
        )
        self.pickle_file_neural_regressor = (
            "agents/UAE_NeuralRegressor_reward/{today_no_dash}/UAE_NeuralRegressor_reward.pkl"
        )

        # catboost regressor
        self.pickle_file_catboost_regressor = (
            "agents/UAE_CatboostRegressor_reward/{today_no_dash}/UAE_CatboostRegressor_reward.pkl"
        )
        self.pickle_file_catboost_regressor_i = (
            "agents/UAE_CatboostRegressor_iReward/{today_no_dash}/UAE_CatboostRegressor_iReward.pkl"
        )

        self.version = version
        self.blob_name = "ace/ace_artifacts/ipm/cmab"

        self.model_config_file_path: Optional[Path] = None

        self.pickle_file_neural_ucb_path: Optional[Path] = None
        self.pickle_file_lin_ucb_path: Optional[Path] = None
        self.pickle_file_lin_ucb_i_path: Optional[Path] = None

        self.pickle_file_neural_regressor_i_path: Optional[Path] = None
        self.pickle_file_neural_regressor_path: Optional[Path] = None

        self.pickle_file_catboost_regressor_path: Optional[Path] = None
        self.pickle_file_catboost_regressor_i_path: Optional[Path] = None

        self.s3_downloader = S3DownloadManager(log_attrs=self._artifacts_id, s3_app_config=s3_app_config)

        self._experiment_agents: dict[VariantName, Any] = {}

    @property
    def agents(self) -> dict[VariantName, Any]:
        return self._experiment_agents

    @cached_property
    def _artifacts_id(self) -> dict:
        return dict(version=self.version)

    @cached_property
    def artifacts_id_repr(self) -> str:
        return f"V={self.version}"

    @classmethod
    def day_with_offset(cls, date_offset: int) -> dt.date:
        return dt.datetime.utcnow().date() - dt.timedelta(days=date_offset)

    @classmethod
    def format_day_with_offset_no_dash(cls, date_offset: int) -> str:
        return cls.day_with_offset(date_offset).strftime("%Y%m%d")

    def get_src_key_path(self, filename: str, date_offset: int):
        formatted_filename = filename.format_map({"today_no_dash": self.format_day_with_offset_no_dash(date_offset)})
        return f"{self.blob_name}/{formatted_filename}"

    def get_dst_file_path(self, filename: str, date_offset: int) -> Path:
        formatted_filename = filename.format_map({"today_no_dash": self.format_day_with_offset_no_dash(date_offset)})
        return self.base_local_dir / formatted_filename

    @newrelic.agent.function_trace()
    @retry(
        retry=retry_if_exception_type(FileNotFoundError),  # retry if S3 file not found
        stop=stop_after_attempt(20),  # retry 9 time
        wait=wait_fixed(0.1),  # wait 100 milliseconds between retries
        before_sleep=fallback_model_logic,
    )
    async def download_model_artifacts(self, date_offset: int):
        try:
            files_were_reloaded = await self._download_model_artifacts(date_offset)
            return files_were_reloaded, date_offset
        except FileNotFoundError as ex:
            logger.warning(
                f"File not found on S3 corresponding to today date with offset={date_offset}: "
                f"{self.format_day_with_offset_no_dash(date_offset)}. Error: {ex}"
            )
            if not self._experiment_agents:
                raise  # no agent was loaded, then need to find at least something
            logger.info("Will continue using pre loaded models from prev days.")

    async def _download_model_artifacts(self, date_offset: int) -> bool:
        model_keys = (
            self.model_config_file,
            self.pickle_file_neural_regressor_i,
            self.pickle_file_neural_regressor,
            # now catboost is optional
            self.pickle_file_catboost_regressor,
            self.pickle_file_catboost_regressor_i,
        )
        keys_and_destinations = [
            (self.get_src_key_path(key, date_offset), self.get_dst_file_path(key, date_offset)) for key in model_keys
        ]
        response = await self.s3_downloader.download_batch(
            description="Downloading model config and pickle files",
            bucket_name=self._bucket_name,
            keys_and_destinations=keys_and_destinations,
        )
        (
            (
                self.model_config_file_path,
                self.pickle_file_neural_regressor_i_path,
                self.pickle_file_neural_regressor_path,
                # now catboost is optional
                self.pickle_file_catboost_regressor_path,
                self.pickle_file_catboost_regressor_i_path,
            ),
            reloaded,
        ) = response
        return reloaded

    def load_model_config(self):
        assert self.model_config_file_path, f"[{self.artifacts_id_repr}] NOT SET: model params file path not set"
        assert (
            self.model_config_file_path.exists()
        ), f"[{self.artifacts_id_repr}] NOT FOUND: model params file {self.model_config_file_path}"
        with open(self.model_config_file_path, "r") as file:
            return yaml.safe_load(file)

    def _load_agent(self, path: Path):
        assert path, f"[{self.artifacts_id_repr}] NOT SET: pickle file 1 path not set"
        assert path.exists(), f"[{self.artifacts_id_repr}] NOT FOUND: pickle file {path}"
        with open(path, "rb") as f:
            return pickle.load(f)

    def load_neural_regressor_i_agent(self):
        agent = self._load_agent(self.pickle_file_neural_regressor_i_path)
        return agent

    def load_neural_regressor_agent(self):
        agent = self._load_agent(self.pickle_file_neural_regressor_path)
        return agent

    def load_catboost_regressor_i_agent(self):
        return self._load_agent(self.pickle_file_catboost_regressor_i_path)

    def load_catboost_regressor_agent(self):
        return self._load_agent(self.pickle_file_catboost_regressor_path)

    def load_neural_ucb_agent(self):
        return self._load_agent(self.pickle_file_neural_ucb_path)

    def load_lin_ucb_agent(self):
        return self._load_agent(self.pickle_file_lin_ucb_path)

    def load_lin_ucb_i_agent(self):
        return self._load_agent(self.pickle_file_lin_ucb_i_path)

    def reload_models(self):
        model_params = self.load_model_config()
        # neural_regressor_i_agent = self.load_neural_regressor_i_agent()
        # neural_regressor_agent = self.load_neural_regressor_agent()

        # now catboost is optional
        catboost_regressor_i_agent = self.load_catboost_regressor_i_agent()
        catboost_regressor_agent = self.load_catboost_regressor_agent()

        # Variants' names are a part of the contract between HomeScreen and Ace.
        # Changing that names can break integration with HomeScreen.
        variants_to_agents = (
            (VariantName.REFINED_REWARDS_RL, catboost_regressor_i_agent),
            (VariantName.UPLIFT_MODEL_RL, catboost_regressor_agent),
            (VariantName.NEURAL_NETWORKS, None),
        )

        for variant_name, agent in variants_to_agents:
            if not agent:
                logger.warning(f"Skip agent for variant={variant_name.value}")
                continue
            self._experiment_agents[variant_name] = agent
            self._experiment_agents[variant_name].set_config(model_params, "UAE")
            logger.info(f"Registered agent for variant={variant_name.value}, last_updated={agent.last_updated}")
