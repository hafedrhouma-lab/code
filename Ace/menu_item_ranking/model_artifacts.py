from pathlib import Path

import newrelic.agent
from keras import Sequential

from abstract_ranking.two_tower import TTVersion
from abstract_ranking.two_tower.model_artifacts import ArtefactsManager, S3File, S3FileBatch
from ace.configs.config import AppS3Config
from menu_item_ranking.two_tower.user_models.item_v1 import (
    create_tt_v1_menu_user_model_from_config
)
from menu_item_ranking.two_tower.user_models.item_v2_big import (
    create_tt_v101_menu_user_model_from_config
)


class MenuItemArtefactsManager(ArtefactsManager):
    def __init__(self, s3_app_config: AppS3Config, recall: int, base_dir: str | Path, country: str, version: TTVersion):
        super().__init__(s3_app_config, recall, base_dir, country, version)

        self.menuitem_embeddings_file = S3File(
            s3_file_key=f"tt_item_embeddings_recall@10_{recall}.parquet"
        )
        self.menuitem_features_file = S3File(
            s3_file_key=f"tt_item_features_recall@10_{recall}.parquet"
        )

    @newrelic.agent.function_trace()
    async def download_model_artifacts(self) -> list[bool]:
        return await super().download_model_artifacts([
            S3FileBatch(
                description="menu item embeddings and features",
                s3_files=[
                    self.menuitem_embeddings_file,
                    self.menuitem_features_file,
                ]
            )
        ])

    def _get_s3_base_path(self, version: TTVersion) -> Path:
        if version in TTVersion.MENUITEM_V1:
            return Path(f"ace/ace_artifacts/ranking/twotower_item_{version}")

        raise ValueError(f"Unsupported model version {version}")

    def instantiate_user_model(self) -> (Sequential, set[str]):
        model_params = self.load_user_model_params()
        if self.version is TTVersion.MENUITEM_V1:
            user_model = create_tt_v1_menu_user_model_from_config(**model_params)
        elif self.version is TTVersion.MENUITEM_V2_BIG:
            user_model = create_tt_v101_menu_user_model_from_config(**model_params)
        else:
            raise ValueError(f"Unsupported model version {self.version}")

        model_filepath = self.get_model_weights_file_name()
        user_model.load_weights(model_filepath)
        return user_model, set(model_params["query_features"])
