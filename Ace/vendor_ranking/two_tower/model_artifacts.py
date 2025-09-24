from pathlib import Path
from typing import TYPE_CHECKING

import newrelic.agent
import pandas
from keras import Sequential

from abstract_ranking.two_tower import TTVersion
from abstract_ranking.two_tower.model_artifacts import ArtefactsManager, S3FileBatch, S3File
from ace.configs.config import AppS3Config
from vendor_ranking.two_tower.user_models.v1 import create_v1_user_model_from_config
from vendor_ranking.two_tower.user_models.v2 import create_tt_v2_user_model_from_config
from vendor_ranking.two_tower.user_models.v3 import create_tt_v3_user_model_from_config

if TYPE_CHECKING:
    import tensorflow as tf


class VendorArtifactsManager(ArtefactsManager):
    def __init__(self, s3_app_config: AppS3Config, recall: int, base_dir: str | Path, country: str, version: TTVersion):

        super().__init__(s3_app_config, recall, base_dir, country, version)

        self.ese_embeddings_file = S3File(s3_file_key=f"ese_chains_embeddings_{recall}.parquet")

    def _get_s3_base_path(self, version: TTVersion) -> Path:
        if version in (TTVersion.V2, TTVersion.V22):
            return Path(f"twotower_{version}")
        if version in (TTVersion.V23, TTVersion.V3):
            return Path(f"ace/ace_artifacts/ranking/twotower_{version}")

        raise ValueError(f"Unsupported model version {version}")

    def load_ese_initial_weights(self):
        ese_embeddings_df = pandas.read_parquet(self.ese_embeddings_file.local_file_path)
        return ese_embeddings_df

    @newrelic.agent.function_trace()
    async def download_model_artifacts(self) -> list[bool]:
        additional_files_to_download = list()

        if self.version in (TTVersion.V2, TTVersion.V22, TTVersion.V23, TTVersion.V3):
            additional_files_to_download.append(S3FileBatch(
                description="ese chain embeddings initial weights",
                s3_files=[self.ese_embeddings_file]
            ))
        return await super().download_model_artifacts(additional_files_to_download)

    def instantiate_user_model(self) -> (Sequential, set[str]):
        model_params: dict = self.load_user_model_params()
        if self.version is TTVersion.V1:
            user_model = create_v1_user_model_from_config(**model_params)
        elif self.version in (TTVersion.V2, TTVersion.V22, TTVersion.V23, TTVersion.V3):
            model_params["ese_embeddings_df"] = self.load_ese_initial_weights()
            if self.version in (TTVersion.V2, TTVersion.V22, TTVersion.V23):
                user_model: "tf.keras.Sequential" = create_tt_v2_user_model_from_config(**model_params)
            elif self.version is TTVersion.V3:
                user_model = create_tt_v3_user_model_from_config(**model_params)
            else:
                raise ValueError(f"Unsupported user model version {self.version}")
        else:
            raise ValueError(f"Unsupported user model version {self.version}")

        weights_files_path = self.user_model_weights_file_data.local_file_path.with_suffix("")
        user_model.load_weights(weights_files_path)
        return user_model, set(model_params["query_features"])
