import newrelic.agent
import numpy as np
import polars as pl
import structlog

import ace
from vendor_ranking import SERVICE_NAME, data
from vendor_ranking.two_tower.logic import LogicTwoTowersV22
from abstract_ranking.two_tower.names import (
    VENDOR_ID,
    COSINE_SIMILARITY,
    OFFLINE_AVERAGE_THRESHOLD,
)

log = structlog.get_logger()


@ace.asyncio.to_thread
@newrelic.agent.function_trace()
def apply_kpp_boosting(
    kpp_filtered_vendors: list[int], vendors: list[int]
) -> tuple[int, list[int]]:
    RANK = "rank"
    df = pl.DataFrame({RANK: list(range(1, len(vendors) + 1)), VENDOR_ID: vendors})

    # will use the same ranking penalties matrix for boosting
    matrix = data.ranking_penalties_matrix
    matrix_height = data.ranking_penalties_matrix_height
    matrix_last_row = data.ranking_penalties_matrix[-1]

    features = df.join(matrix, how="left", on="rank")

    features = features.with_columns(
        vendors_rank=pl.when(~pl.col(VENDOR_ID).is_in(kpp_filtered_vendors))
        .then(pl.col(RANK))
        .when(
            (pl.col(RANK) > matrix_height)
            & (pl.col(VENDOR_ID).is_in(kpp_filtered_vendors))
        )
        .then(pl.col(RANK) - matrix_last_row["high_score_penalty"] - 0.1)
        .when(
            (pl.col(RANK) <= matrix_height)
            & (pl.col(VENDOR_ID).is_in(kpp_filtered_vendors))
        )
        .then(pl.col(RANK) - pl.col("high_score_penalty") - 0.1)
    )

    features = features.sort("vendors_rank", nulls_last=True, descending=False)
    num_changed_ranks = features.filter(pl.col("rank") != pl.col("vendors_rank")).height

    return num_changed_ranks, features[VENDOR_ID].to_list()


class KeyPreferredPartner(LogicTwoTowersV22):
    """
    Key Preferred Partner
    Boost KPP champions

    live analytics (ss and std_dev) --> calculated live within req lifetime
    offline analytics (ss and std_dev) --> calculated offline and stored in kpp_vendor_scores table in postgres db
    """

    NAME = SERVICE_NAME + ":kpp_two_towers_v2"
    MODEL_TAG = SERVICE_NAME + ":kpp_two_towers_v2"

    STANDARD_DEVIATION_CONSTANT_LIVE: int = 0
    STANDARD_DEVIATION_CONSTANT_OFFLINE: int = 0
    KPP_VENDORS_LIMIT: int = 10

    _vendors_avg_threshold_live: float = 0.0
    _filtered_kpp_vendors: pl.DataFrame = None

    def _extract_kpp_vendors(self):
        self.exec_log.metadata[
            "sorted_vendors_similarity_scores"
        ] = self.sorted_vendors_similarity_scores[COSINE_SIMILARITY].to_list()
        self._filtered_kpp_vendors = self.sorted_vendors_similarity_scores.join(
            data.kpp_vendor_scores, how="inner", on=VENDOR_ID
        )
        self.stats += [
            (
                "kpp_tt.request.kpp_vendors_found_in_req_cnt",
                len(self._filtered_kpp_vendors),
            )
        ]

    def _compare_vendors_ss_with_threshold_offline(self):
        self._filtered_kpp_vendors = self._filtered_kpp_vendors.filter(
            (
                pl.col(OFFLINE_AVERAGE_THRESHOLD).is_not_null()
                & (pl.col(COSINE_SIMILARITY) > (pl.col(OFFLINE_AVERAGE_THRESHOLD)))
            )
            | (
                pl.col(OFFLINE_AVERAGE_THRESHOLD).is_null()
                & (pl.col(COSINE_SIMILARITY) > (data.kpp_overall_avg_threshold))
            )
        )
        self.exec_log.metadata[
            "kpp_vendor_ss_distribution_threshold_met"
        ] = self._filtered_kpp_vendors[VENDOR_ID].to_list()
        self.stats += [
            (
                "kpp_tt.request.offline_threshold_met_vendors_cnt",
                len(self._filtered_kpp_vendors),
            )
        ]

    def _calculate_threshold_live(self):
        vendors_avg_similarity_score_live = self.sorted_vendors_similarity_scores[
            COSINE_SIMILARITY
        ].mean()
        vendors_std_dev_ss_live = np.nanstd(
            self.sorted_vendors_similarity_scores[COSINE_SIMILARITY]
        )
        self._vendors_avg_threshold_live = vendors_avg_similarity_score_live + (
            self.STANDARD_DEVIATION_CONSTANT_LIVE * vendors_std_dev_ss_live
        )
        self.exec_log.metadata["computed_live_threshold"] = float(
            self._vendors_avg_threshold_live
        )
        self.stats += [
            (
                "kpp_tt.request.computed_live_threshold",
                float(self._vendors_avg_threshold_live),
            )
        ]

    def _compare_kpp_vendors_with_threshold_live(self):
        self._filtered_kpp_vendors = self._filtered_kpp_vendors.filter(
            pl.col(COSINE_SIMILARITY) > self._vendors_avg_threshold_live
        )
        self.exec_log.metadata[
            "request_ss_distribution_threshold_met"
        ] = self._filtered_kpp_vendors[VENDOR_ID].to_list()
        self.stats += [
            (
                "kpp_tt.request.live_threshold_met_vendors_cnt",
                len(self._filtered_kpp_vendors),
            )
        ]

    def _prepare_kpp_vendors(self) -> list[int]:
        # 1) extract all kpp vendors from tt output
        self._extract_kpp_vendors()
        # 2) compare vendors similarity score with kpp table (offline threshold)
        self._compare_vendors_ss_with_threshold_offline()
        # 3) calculate threshold live = avg_req_all_vendors_ss + n std_dev_req_all_vendors_ss
        self._calculate_threshold_live()
        # 4) compare kpp vendors ss with threshold live
        self._compare_kpp_vendors_with_threshold_live()
        # 5) limit number of kpp vendors to KPP_VENDORS_LIMIT
        filtered_kpp_vendors_list = self._filtered_kpp_vendors[VENDOR_ID].to_list()
        filtered_kpp_vendors_list = filtered_kpp_vendors_list[: self.KPP_VENDORS_LIMIT]
        return filtered_kpp_vendors_list

    @newrelic.agent.function_trace()
    async def sort(self) -> list[int]:
        tt_sorted_vendors_list = await super().sort()
        filtered_kpp_vendors_list = self._prepare_kpp_vendors()
        num_changed_ranks, final_sorting = await apply_kpp_boosting(
            filtered_kpp_vendors_list, tt_sorted_vendors_list
        )

        self.stats += [("kpp_tt.request.num_changed_ranks", num_changed_ranks)]

        self.exec_log.metadata["perseusSessionId"] = self.request.perseusSessionId
        self.exec_log.metadata[
            "offline_standard_deviation_constant"
        ] = self.STANDARD_DEVIATION_CONSTANT_OFFLINE
        self.exec_log.metadata[
            "live_standard_deviation_constant"
        ] = self.STANDARD_DEVIATION_CONSTANT_LIVE
        self.exec_log.metadata["two_tower_sorting_output"] = tt_sorted_vendors_list

        return final_sorting
