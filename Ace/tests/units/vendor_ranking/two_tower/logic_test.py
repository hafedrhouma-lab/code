import math
from pathlib import Path
from unittest import mock
from unittest.mock import AsyncMock, Mock

import numpy
import numpy as np
import pandas as pd
import polars as pl
import pydantic
import pytest
from pydantic import ValidationError

from ace.model_log import LogEntry
from vendor_ranking import SERVICE_NAME
from vendor_ranking.context import Context
from vendor_ranking.input import VendorList
from abstract_ranking.two_tower import TTVersion
from vendor_ranking.two_tower.logic import (
    LogicTwoTowersV22,
    infer_user_embedding_two_tower,
)
from vendor_ranking.two_tower.model_artifacts import VendorArtifactsManager
from vendor_ranking.two_tower.repository.models import (
    UserStaticFeaturesV2,
    UserDynamicFeatures,
)


def get_features_and_embeddings_paths(
    country_code: str, recall: int, version: TTVersion
) -> (Path, Path):
    if version in (TTVersion.V23, TTVersion.V3):
        base_path = f"tests/fixtures/s3/ace/ace_artifacts/ranking/twotower_{version}"
    elif version in (TTVersion.V2, TTVersion.V22):
        base_path = f"tests/fixtures/s3/twotower_{version}"
    else:
        raise ValueError(f"Unsupported version {version}")

    path = f"{base_path}/{country_code.upper()}/model_artifacts_recall_{recall}"
    user_features_path = Path(f"{path}/streamlit_{recall}_test_df.parquet")
    assert user_features_path.exists(), f"path not found: {user_features_path}"

    user_embeddings_path = Path(
        f"{path}/tt_user_embeddings_recall@10_{recall}.parquet"
    )
    assert user_embeddings_path.exists(), f"path not found: {user_embeddings_path}"

    return user_features_path, user_embeddings_path


@pytest.fixture
def two_towers_request_body() -> dict:
    fixture_path = Path(
        "tests/fixtures/vendor_ranking/two_towers_ranking/v2/request.json"
    )
    assert fixture_path.exists()
    return pydantic.parse_file_as(dict, fixture_path)


@pytest.fixture
def two_towers_request(two_towers_request_body: dict) -> VendorList:
    return pydantic.parse_obj_as(VendorList, two_towers_request_body)


@pytest.fixture
def two_towers_unittest_request() -> VendorList:
    fixture_path = Path(
        "tests/fixtures/vendor_ranking/two_towers_ranking/v2/request.json"
    )
    assert fixture_path.exists()
    return pydantic.parse_file_as(VendorList, fixture_path)


@pytest.fixture
def two_towers_unittest_response() -> list[int]:
    fixture_path = Path(
        "tests/fixtures/vendor_ranking/two_towers_ranking/v2/response.json"
    )
    assert fixture_path.exists()
    return pydantic.parse_file_as(list[int], fixture_path)


@pytest.fixture()
def user_static_features(two_towers_unittest_request) -> UserStaticFeaturesV2:
    if two_towers_unittest_request.customer_id < 0:
        return UserStaticFeaturesV2(
            account_id=two_towers_unittest_request.customer_id,
            country_iso=two_towers_unittest_request.location.country_code,
            most_recent_10_clicks_wo_orders="no_recent_clicks",
            most_recent_10_orders="first_order",
            frequent_clicks="no_frequent_clicks",
            frequent_chains="no_frequent_orders",
            most_recent_15_search_keywords="no_prev_search",
        )

    return UserStaticFeaturesV2(
        account_id=two_towers_unittest_request.customer_id,
        country_iso=two_towers_unittest_request.location.country_code,
        most_recent_10_clicks_wo_orders=(
            "649666 644518 645014 637612 637645 637643 627138 607598 651767 12925 648630 644471 18721 16810 18918"
        ),
        most_recent_10_orders="620125 631280 658577 17853 628893 662372 1863 1826 1968 636829",
        frequent_clicks="12925 1979 7614 633253 10329 649666 16040 637612 660247 "
        "645014 648630 5699 619320 642296 20968",
        frequent_chains="637934 635527 628893 635646 1968 636829 653826 631280",
        most_recent_15_search_keywords=(
            "paul sandwitch Sandwiches chicken roasted baha crave buffalo burger "
            "grilled chicken manoushe starbucks kfc bowl starbucks starbucks qunioa"
        ),
    )


@pytest.fixture()
def user_dynamic_features(two_towers_unittest_request) -> UserDynamicFeatures:
    return UserDynamicFeatures.from_request(two_towers_unittest_request)


@pytest.fixture()
def static_and_dynamic_features_names() -> list[str]:
    names: set[str] = (
        UserStaticFeaturesV2.schema(by_alias=True)["properties"].keys()
        | UserDynamicFeatures.schema(by_alias=False)["properties"].keys()
    )
    return list(names)


@pytest.fixture()
def two_towers_logic(
    app_context: "Context",
    two_towers_unittest_request,
    user_static_features: UserStaticFeaturesV2,
    user_dynamic_features: UserDynamicFeatures,
):
    get_user_static_features_mock = AsyncMock()
    get_user_static_features_mock.return_value = user_static_features

    get_user_dynamic_features_two_tower_mock = Mock()
    get_user_dynamic_features_two_tower_mock.return_value = user_dynamic_features

    with mock.patch(
        "vendor_ranking.two_tower.repository.users.TwoTowersUsersRepository.get_user_static_features_two_tower",
        new=get_user_static_features_mock,
    ):
        logic = LogicTwoTowersV22(
            request=two_towers_unittest_request,
            artifacts_service_registry=app_context.artifacts_service_registry,
            exec_log=LogEntry(SERVICE_NAME, two_towers_unittest_request.dict()),
        )
        yield logic


# noinspection PyMethodMayBeStatic
class LogicTwoTowersTest:
    @pytest.mark.asyncio
    async def test_get_customer_embeddings(
        self, two_towers_logic: LogicTwoTowersV22, two_towers_request: VendorList
    ):
        inferred_customers_embedding: numpy.ndarray = (
            await two_towers_logic._get_customer_embedding(request=two_towers_request)
        )
        embedding = list(inferred_customers_embedding)
        assert all(
            [
                math.isclose(float(current), expected, rel_tol=1e-05)
                for current, expected in zip(
                    embedding,
                    [
                        -0.10113396,
                        0.20435944,
                        -0.06167142,
                        0.17128094,
                        0.23207812,
                        -0.12593386,
                        -0.11113324,
                        -0.046112113,
                        0.20275089,
                        0.16050522,
                        -0.23665537,
                        -0.2687063,
                        -0.0053510945,
                        0.15780273,
                        -0.36285025,
                        -0.04519448,
                        0.056479037,
                        0.410956,
                        -0.18545073,
                        -0.017043151,
                        0.26043063,
                        0.12929316,
                        0.020760318,
                        -0.050153304,
                        0.24470747,
                        -0.035313558,
                        0.0639248,
                        0.038040698,
                        -0.24846792,
                        -0.022661384,
                        -0.20127884,
                        -0.11528474,
                    ],
                )
            ]
        )

    @pytest.mark.asyncio
    async def test_prepare_features_base(self, two_towers_logic: LogicTwoTowersV22):
        df: "pl.DataFrame" = await two_towers_logic.prepare_features()
        assert df.shape[1] == 12, "wrong features count"

    @pytest.mark.asyncio
    async def test_prepare_features_unknown_country(
        self, two_towers_logic: LogicTwoTowersV22, monkeypatch
    ):
        monkeypatch.setattr(
            two_towers_logic.request.location, "country_code", "UNKNOWN_COUNTRY"
        )
        with pytest.raises(ValidationError):
            await two_towers_logic.prepare_features()

    @pytest.mark.asyncio
    async def test_sort_vendors(
        self,
        two_towers_logic: LogicTwoTowersV22,
        two_towers_unittest_response: list[int],
    ):
        await two_towers_logic.prepare_features()
        ranked_vendors: list[int] = await two_towers_logic.sort()
        assert ranked_vendors == two_towers_unittest_response

    @pytest.mark.parametrize(
        "country_code, recall, version",
        [
            ("BH", 4820, TTVersion.V3),
            ("OM", 5169, TTVersion.V3),
            ("KW", 4926, TTVersion.V3),
            ("QA", 4705, TTVersion.V2),
            ("AE", 4858, TTVersion.V2),
            ("AE", 4566, TTVersion.V22),
            ("BH", 4718, TTVersion.V22),
            ("QA", 4632, TTVersion.V22),
            ("BH", 4695, TTVersion.V23),
            ("OM", 4991, TTVersion.V23),
        ],
    )
    @pytest.mark.asyncio
    async def test_two_tower_model_correctness(
        self,
        app_context: "Context",
        country_code: str,
        recall: int,
        version: TTVersion,
        tmp_path: "Path",
    ):
        user_features_path, user_embeddings_path = get_features_and_embeddings_paths(
            country_code, recall, version=version
        )

        all_user_features_df = pd.read_parquet(user_features_path)

        expected_user_embeddings_df = pd.read_parquet(user_embeddings_path)
        expected_embedding = np.array(
            [a for a in expected_user_embeddings_df.user_embeddings.values]
        )

        artifacts_service = app_context.artifacts_service_registry.get(version)
        artifacts_manager = VendorArtifactsManager(
            s3_app_config=artifacts_service.s3_app_config,
            recall=recall,
            base_dir=tmp_path / f"twotower_{version}_{country_code}_{recall}",
            country=country_code,
            version=version,
        )
        await artifacts_manager.download_model_artifacts()
        user_model, features_names = artifacts_manager.instantiate_user_model()
        user_features_df = all_user_features_df[list(features_names)]
        inferred_user_embedding = infer_user_embedding_two_tower(
            features=user_features_df, user_model=user_model
        )

        assert np.allclose(expected_embedding[0], inferred_user_embedding, atol=1.0e-4)
