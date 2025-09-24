from datetime import datetime

import polars as pl
import pytest

from ace.storage import db
from vendor_ranking import data
from vendor_ranking.data import (
    refresh_vendors,
    refresh_ranking_penalties_v2,
    get_vendors,
    select_score,
    get_all_vendors,
)


@pytest.mark.asyncio
async def test_refresh_vendors(setup_db):
    await refresh_vendors()
    assert get_all_vendors() is not None


@pytest.mark.asyncio
async def test_refresh_ranking_penalties_v2(setup_db):
    await refresh_ranking_penalties_v2()
    assert data.ranking_penalties_matrix_v2 is not None
    assert data.vendor_penalty_scores_v2 is not None


@pytest.mark.asyncio
async def test_refresh_ranking_penalties_v2_failure():
    """No db initialization"""
    try:
        await refresh_ranking_penalties_v2()
    except Exception as e:
        assert e is AssertionError
        assert data.ranking_penalties_matrix_v2 is None
        assert data.vendor_penalty_scores_v2 is None


@pytest.mark.asyncio
async def test_get_vendors(setup_db):
    country_id = 4
    vendors = pl.Series("vendor_id", [699654, 683052, 41895, 679355])

    await refresh_vendors()

    response = get_vendors(country_id, vendors, datetime.now())
    num_of_rows = response.collect().shape[0]
    assert num_of_rows == len(vendors)
    assert "timeofday_score" in response


def test_select_score():
    assert select_score(0) == "_1_midnight_snack_score"
    assert select_score(6) == "_1_midnight_snack_score"
    assert select_score(7) == "_2_breakfast_score"
    assert select_score(11) == "_2_breakfast_score"
    assert select_score(12) == "_3_lunch_score"
    assert select_score(15) == "_3_lunch_score"
    assert select_score(16) == "_4_evening_snack_score"
    assert select_score(18) == "_4_evening_snack_score"
    assert select_score(19) == "_5_dinner_score"
    assert select_score(23) == "_5_dinner_score"
