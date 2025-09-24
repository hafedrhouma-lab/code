import datetime as dt
from unittest import mock
from unittest.mock import MagicMock

import pytest
from httpx import AsyncClient

from nba.app import refresh_models, AGENTS_MANAGER
from nba.data import refresh_banners_inference_data


@pytest.mark.asyncio
async def test_nba_refresh_banners_data(nba_client: AsyncClient):
    banners_df, banners_date = await refresh_banners_inference_data()
    assert banners_df is not None and not banners_df.empty, "banners weren't loaded"
    assert banners_date, "banners latest snapshot data is unknown"


@pytest.mark.asyncio
async def test_nba_refresh_models_data(nba_client: AsyncClient):
    datetime_mock = MagicMock()
    datetime_mock.return_value = dt.datetime(year=2023, month=10, day=11)

    with mock.patch("nba.hero_banner.agents.AgentsManager.day_with_offset", new=datetime_mock):
        artifacts_latest_date = await refresh_models()
        assert artifacts_latest_date
        assert AGENTS_MANAGER.pickle_file_neural_regressor_i_path
        assert AGENTS_MANAGER.pickle_file_neural_regressor_path
