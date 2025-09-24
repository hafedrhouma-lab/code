from typing import TYPE_CHECKING

import pytest

from menu_item_ranking.model_artifacts import MenuItemArtefactsManager

if TYPE_CHECKING:
    pass


# noinspection PyMethodMayBeStatic
class MenuItemArtifactsManagerTest:
    @pytest.mark.asyncio
    async def test_s3_menu_item_download_model_artifacts(
        self, menu_item_artifacts_manager: "MenuItemArtefactsManager"
    ):
        reloaded = await menu_item_artifacts_manager.download_model_artifacts()
        assert all(item is True for item in reloaded)
        for path in (
            menu_item_artifacts_manager.user_model_weights_file_data.local_file_path,
            menu_item_artifacts_manager.user_model_params_file.local_file_path,
            menu_item_artifacts_manager.menuitem_embeddings_file.local_file_path,
            menu_item_artifacts_manager.menuitem_features_file.local_file_path
        ):
            assert path, "path value is not initialized"
            assert path.exists(), f"downloaded file not found: {path}"
        reloaded = await menu_item_artifacts_manager.download_model_artifacts()
        assert all(item is False for item in reloaded)
        for path in (
            menu_item_artifacts_manager.user_model_weights_file_data.local_file_path,
            menu_item_artifacts_manager.user_model_params_file.local_file_path,
            menu_item_artifacts_manager.menuitem_embeddings_file.local_file_path,
            menu_item_artifacts_manager.menuitem_features_file.local_file_path
        ):
            assert path, "path value is not initialized"
            assert path.exists(), f"downloaded file not found: {path}"

    @pytest.mark.asyncio
    async def test_menu_item_instantiate_user_model(
        self, menu_item_artifacts_manager: "MenuItemArtefactsManager"
    ):
        await menu_item_artifacts_manager.download_model_artifacts()
        user_model, _ = menu_item_artifacts_manager.instantiate_user_model()
        names = [a.name for a in user_model.layers[0].layers]
        assert "delivery_area_id_layer" in names
        assert "order_hour_layer" in names
