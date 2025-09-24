from typing import Awaitable

import newrelic.agent
import polars as pl

import ace
from vendor_ranking import data, SERVICE_NAME, personalized_ranking
from abstract_ranking.two_tower import TTVersion
from vendor_ranking.two_tower.logic import (
    LogicTwoTowersV22,
    LogicTwoTowersV3,
    LogicTwoTowersV23,
)


@ace.asyncio.to_thread
@newrelic.agent.function_trace()
def apply_ranking_penalty(vendors: list[int]) -> Awaitable[list[int]]:
    df = pl.DataFrame({"rank": list(range(1, len(vendors) + 1)), "vendor_id": vendors})

    scores = data.vendor_penalty_scores
    matrix = data.ranking_penalties_matrix
    matrix_height = data.ranking_penalties_matrix_height
    matrix_last_row = data.ranking_penalties_matrix[-1]

    features = df.join(scores, how="left", on="vendor_id").join(
        matrix, how="left", on="rank"
    )

    features = features.with_columns(
        vendors_rank=pl.when(pl.col("penalty_bin").is_null())
        .then(pl.col("rank"))
        .when(
            (pl.col("rank") > matrix_height)
            & (pl.col("penalty_bin") == "low_score_penalty")
        )
        .then(pl.col("rank") + matrix_last_row["low_score_penalty"] + 0.1)
        .when(
            (pl.col("rank") > matrix_height)
            & (pl.col("penalty_bin") == "medium_score_penalty")
        )
        .then(pl.col("rank") + matrix_last_row["medium_score_penalty"] + 0.1)
        .when(
            (pl.col("rank") > matrix_height)
            & (pl.col("penalty_bin") == "high_score_penalty")
        )
        .then(pl.col("rank") + matrix_last_row["high_score_penalty"] + 0.1)
        .when(
            (pl.col("rank") <= matrix_height)
            & (pl.col("penalty_bin") == "low_score_penalty")
        )
        .then(pl.col("rank") + pl.col("low_score_penalty") + 0.1)
        .when(
            (pl.col("rank") <= matrix_height)
            & (pl.col("penalty_bin") == "medium_score_penalty")
        )
        .then(pl.col("rank") + pl.col("medium_score_penalty") + 0.1)
        .when(
            (pl.col("rank") <= matrix_height)
            & (pl.col("penalty_bin") == "high_score_penalty")
        )
        .then(pl.col("rank") + pl.col("high_score_penalty") + 0.1)
    )

    features = features.sort("vendors_rank", nulls_last=True, descending=False)

    return features["vendor_id"].to_list()


class Logic(personalized_ranking.Logic):
    """CatBoost
    FastSort - YES
    Penalization - YES
    """

    NAME = SERVICE_NAME + ":price_parity"
    NICKNAME: str = "price_parity"

    async def sort(self) -> list[int]:
        final_sorting = await super().sort()
        final_sorting = await apply_ranking_penalty(final_sorting)

        return final_sorting


class TTv2PriceParity(LogicTwoTowersV22):
    """TwoTowers.
    Penalization - YES.
    """

    VERSION: TTVersion = LogicTwoTowersV22.VERSION
    NAME: str = f"{SERVICE_NAME}:tt_{VERSION}_price_parity"
    NICKNAME: str = f"tt_{VERSION}_price_parity"
    MODEL_TAG: str = f"{SERVICE_NAME}:tt_{VERSION}_price_parity_20231108"

    async def sort(self) -> list[int]:
        final_sorting = await super().sort()  # tow towers
        final_sorting = await apply_ranking_penalty(final_sorting)  # price parity

        return final_sorting


class TTv23PriceParity(LogicTwoTowersV23):
    """TwoTowers V23 + KPP"""

    VERSION: TTVersion = LogicTwoTowersV23.VERSION
    NAME: str = f"{SERVICE_NAME}:tt_{VERSION}_price_parity"
    NICKNAME: str = f"tt_{VERSION}_price_parity"
    MODEL_TAG: str = f"{SERVICE_NAME}:tt_{VERSION}_price_parity_20231214"

    async def sort(self) -> list[int]:
        final_sorting = await super().sort()  # tow towers
        final_sorting = await apply_ranking_penalty(final_sorting)  # price parity

        return final_sorting


class TTv3PriceParity(LogicTwoTowersV3):
    """TwoTowers V3 + KPP"""

    VERSION: TTVersion = LogicTwoTowersV3.VERSION
    NAME: str = f"{SERVICE_NAME}:tt_{VERSION}_price_parity"
    NICKNAME: str = f"tt_{VERSION}_price_parity"
    MODEL_TAG: str = f"{SERVICE_NAME}:tt_{VERSION}_price_parity_20231214"

    async def sort(self) -> list[int]:
        final_sorting = await super().sort()  # tow towers
        final_sorting = await apply_ranking_penalty(final_sorting)  # price parity

        return final_sorting
