import asyncio

import newrelic.agent
import polars as pl

import ace
from ace.storage.db import assert_table_exists
from vendor_ranking import data, SERVICE_NAME
from abstract_ranking.two_tower import TTVersion
from vendor_ranking.two_tower.logic import LogicTwoTowersV22
from vendor_ranking.price_parity_v2.names import (
    VENDOR_ID,
    RANK_COL,
    PENALTY_BIN_COL,
    BIN_1,
    BIN_2,
    BIN_3,
    BIN_4,
    BIN_5,
    BIN_6,
    BIN_7,
    BIN_8,
    BIN_9,
    BIN_10,
)


@ace.asyncio.to_thread
@newrelic.agent.function_trace()
def apply_ranking_penalty(vendors: list[int]) -> tuple[int, list[int]]:
    df = pl.DataFrame(
        {
            RANK_COL: list(range(1, len(vendors) + 1)),
            VENDOR_ID: vendors,
        }
    )

    scores, matrix, matrix_height = data.get_pp_v2_scoring()
    matrix_last_row = matrix[-1]

    features = df.join(scores, how="left", on=VENDOR_ID).join(
        matrix, how="left", on=RANK_COL
    )

    rank_col = pl.col(RANK_COL)
    penalty_bit_col = pl.col(PENALTY_BIN_COL)

    features = features.with_columns(
        vendors_rank=pl.when(penalty_bit_col.is_null())
        .then(rank_col)
        .when((rank_col > matrix_height) & (penalty_bit_col == BIN_1))
        .then(rank_col + matrix_last_row[BIN_1] + 0.1)
        .when((rank_col > matrix_height) & (penalty_bit_col == BIN_2))
        .then(rank_col + matrix_last_row[BIN_2] + 0.1)
        .when((rank_col > matrix_height) & (penalty_bit_col == BIN_3))
        .then(rank_col + matrix_last_row[BIN_3] + 0.1)
        .when((rank_col > matrix_height) & (penalty_bit_col == BIN_4))
        .then(rank_col + matrix_last_row[BIN_4] + 0.1)
        .when((rank_col > matrix_height) & (penalty_bit_col == BIN_5))
        .then(rank_col + matrix_last_row[BIN_5] + 0.1)
        .when((rank_col > matrix_height) & (penalty_bit_col == BIN_6))
        .then(rank_col + matrix_last_row[BIN_6] + 0.1)
        .when((rank_col > matrix_height) & (penalty_bit_col == BIN_7))
        .then(rank_col + matrix_last_row[BIN_7] + 0.1)
        .when((rank_col > matrix_height) & (penalty_bit_col == BIN_8))
        .then(rank_col + matrix_last_row[BIN_8] + 0.1)
        .when((rank_col > matrix_height) & (penalty_bit_col == BIN_9))
        .then(rank_col + matrix_last_row[BIN_9] + 0.1)
        .when((rank_col > matrix_height) & (penalty_bit_col == BIN_10))
        .then(rank_col + matrix_last_row[BIN_10] + 0.1)
        .when((rank_col <= matrix_height) & (penalty_bit_col == BIN_1))
        .then(rank_col + pl.col(BIN_1) + 0.1)
        .when((rank_col <= matrix_height) & (penalty_bit_col == BIN_2))
        .then(rank_col + pl.col(BIN_2) + 0.1)
        .when((rank_col <= matrix_height) & (penalty_bit_col == BIN_3))
        .then(rank_col + pl.col(BIN_3) + 0.1)
        .when((rank_col <= matrix_height) & (penalty_bit_col == BIN_4))
        .then(rank_col + pl.col(BIN_4) + 0.1)
        .when((rank_col <= matrix_height) & (penalty_bit_col == BIN_5))
        .then(rank_col + pl.col(BIN_5) + 0.1)
        .when((rank_col <= matrix_height) & (penalty_bit_col == BIN_6))
        .then(rank_col + pl.col(BIN_6) + 0.1)
        .when((rank_col <= matrix_height) & (penalty_bit_col == BIN_7))
        .then(rank_col + pl.col(BIN_7) + 0.1)
        .when((rank_col <= matrix_height) & (penalty_bit_col == BIN_8))
        .then(rank_col + pl.col(BIN_8) + 0.1)
        .when((rank_col <= matrix_height) & (penalty_bit_col == BIN_9))
        .then(rank_col + pl.col(BIN_9) + 0.1)
        .when((rank_col <= matrix_height) & (penalty_bit_col == BIN_10))
        .then(rank_col + pl.col(BIN_10) + 0.1)
    )

    features = features.sort("vendors_rank", nulls_last=True, descending=False)
    num_changed_ranks = features.filter(
        pl.col(RANK_COL) != pl.col("vendors_rank")
    ).height

    return num_changed_ranks, features[VENDOR_ID].to_list()


class Logic(LogicTwoTowersV22):
    """TwoTowers V22.
    Penalization (V2) - YES.
    """

    VERSION: TTVersion = LogicTwoTowersV22.VERSION
    NAME: str = f"{SERVICE_NAME}:tt_{VERSION}_price_parity_v2"
    NICKNAME: str = f"tt_{VERSION}_price_parity_v2"
    MODEL_TAG: str = f"{SERVICE_NAME}:tt_{VERSION}_price_parity_v2_20231220"

    @classmethod
    async def check_execution_requirements(cls, msg: str):
        """
        Postgres DB must contain not empty tables with price parity vendor scores and penalty rank matrix.
        """
        # 1: validate database tables
        await asyncio.gather(
            *(
                assert_table_exists(table_name=table_name, msg=msg)
                for table_name in (
                    data.pp_v2_vendor_scores_table_name,
                    data.pp_v2_ranking_penalties_matrix_table_name,
                )
            )
        )

    async def sort(self) -> list[int]:
        tt_sorting = await super().sort()  # tow towers
        num_changed_ranks, final_sorting = await apply_ranking_penalty(
            tt_sorting
        )  # price parity

        self.stats += [("pp_tt.request.num_changed_ranks", num_changed_ranks)]
        self.exec_log.metadata["two_tower_sorting_output"] = tt_sorting
        self.exec_log.metadata["num_of_changed_ranks"] = num_changed_ranks

        return final_sorting
