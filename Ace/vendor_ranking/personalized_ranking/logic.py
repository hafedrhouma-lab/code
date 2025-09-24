import newrelic.agent
import numpy as np
import polars as pl

import ace.asyncio
from ace import ml
from ace.storage import db
from vendor_ranking import data, SERVICE_NAME
from vendor_ranking.base_logic import VendorBaseLogic
from vendor_ranking.input import VendorFeatures, split_by_vendor_availability
from vendor_ranking.personalized_ranking import model as catboost_model

CUSTOMER_EMBEDDINGS_QUERY = "SELECT embeddings FROM ese_account_embeddings WHERE account_id = $1"

# By country (positive weight means higher feature value is better)
FAST_SORT_WEIGHTS = {
    # Kuwait
    1: {
        "delivery_fee": -0.038,
        "delivery_time": -0.038,
        "vendor_rating": 0.019,
        "is_tgo": 0.205,
        "popularity_org_score": 0.375,
        "timeofday_score": 0.25,
        "vendor_fail_rate": -0.019,
    },
    # Bahrain
    3: {
        "delivery_fee": -0.224,
        "delivery_time": -0.224,
        "vendor_rating": 0.113,
        "timeofday_score": 0.25,
        "vendor_fail_rate": -0.113,
    },
    # Oman
    5: {
        "delivery_fee": -0.225,
        "delivery_time": -0.135,
        "vendor_rating": 0.09,
        "is_tgo": 0.09,
        "timeofday_score": 0.1,
        "vendor_fail_rate": -0.135,
    },
    # Qatar
    6: {
        "delivery_fee": -0.038,
        "delivery_time": -0.075,
        "vendor_rating": 0.038,
        "popularity_org_score": 0.375,
        "timeofday_score": 0.25,
        "vendor_fail_rate": -0.205,
    },
    # Jordan
    8: {
        "delivery_fee": -0.2,
        "delivery_time": -0.15,
        "vendor_rating": 0.1,
        "min_order_amount": -0.05,
        "is_tgo": 0.15,
        "vendor_fail_rate": -0.2,
    },
    # Egypt
    9: {
        "delivery_fee": -0.056,
        "delivery_time": -0.094,
        "vendor_rating": 0.038,
        "is_tgo": 0.056,
        "popularity_org_score": 0.375,
        "timeofday_score": 0.25,
        "vendor_fail_rate": -0.094,
    },
    # Iraq
    10: {
        "delivery_fee": -0.056,
        "delivery_time": -0.094,
        "vendor_rating": 0.038,
        "popularity_org_score": 0.375,
        "timeofday_score": 0.25,
        "vendor_fail_rate": -0.094,
    },
}
DEFAULT_FAST_SORT_WEIGHTS = {  # KSA / LB / UAE
    "delivery_fee": -0.075,
    "delivery_time": -0.075,
    "is_tgo": 0.15,
    "vendor_rating": 0.056,
    "popularity_org_score": 0.375,
    "timeofday_score": 0.25,
    "vendor_fail_rate": -0.019,
}

EXCHANGE_RATES = {
    1: 0.32601,  # KWD
    2: 3.98797,  # SAR
    3: 0.40013,  # BHD
    4: 3.90278,  # AED
    5: 0.4086,  # OMR
    6: 3.86886,  # QAR
    8: 0.75379,  # JOD
    9: 32.71979,  # EGP
    10: 1551.25114,  # IQD
}


def lc_to_eur(country_id) -> float:
    return EXCHANGE_RATES.get(country_id, 1)  # Better default?..


@newrelic.agent.function_trace()
def _combine_features(dynamic_features: pl.DataFrame, static_features: pl.DataFrame) -> pl.DataFrame:
    features = dynamic_features.join(static_features, how="left", on="vendor_id")
    features = features.select(
        [
            "vendor_id",
            "delivery_fee",
            "delivery_time",
            "vendor_rating",
            "status",
            "min_order_amount",
            "is_tgo",
            "rating_count",
            "popularity_org_score",
            "timeofday_score",
            "new_customer_retention_14",
            "vendor_fail_rate",
            "aov_eur",
            "new_order_prc",
            "promotion_order_prc",
            "menu_to_cart_rate",
            "dwell_time_seconds_mean",
            "cosine_similarity",
            "chain_name",
            "chain_id",
        ]
    )

    return features


class Logic(VendorBaseLogic):
    """CatBoost
    FastSort - YES
    Penalization - NO
    """

    NAME = SERVICE_NAME + ":personalized"
    MODEL_TAG = catboost_model.MODEL_TAG
    NICKNAME = "catboost_with_fast_sort"
    TOP_N_FAST_SORT_LIMIT: int = 200

    _sorted_unavailable_vendors: list[int]
    _features: pl.DataFrame

    @newrelic.agent.function_trace()
    async def _get_customer_embedding(self, customer_id) -> np.ndarray:
        async with db.connections().acquire() as conn:
            return await db.fetch_embedding(conn, CUSTOMER_EMBEDDINGS_QUERY, customer_id)

    def _get_dynamic_features(self) -> pl.DataFrame:
        if self.request.vendors_df:
            df = pl.DataFrame(self.request.vendors_df, schema=VendorFeatures.df_schema, orient="row")
        else:
            df = pl.DataFrame(self.request.vendors, schema=VendorFeatures.df_schema)

        return df

    @ace.asyncio.to_thread
    @newrelic.agent.function_trace()
    def _fast_sort(self, df: pl.DataFrame) -> pl.DataFrame:
        weights = FAST_SORT_WEIGHTS.get(self.request.location.country_id, DEFAULT_FAST_SORT_WEIGHTS)

        feature_columns = list(weights.keys())
        feature_weights = list(weights.values())

        # These columns HAVE to have some non-null values, otherwise an exception will be raised
        null_max_columns = set(feature_columns) & {
            "delivery_fee",
            "delivery_time",
            "min_order_amount",
        }

        features_df = (
            df.lazy()
            .select(["vendor_id"] + feature_columns)
            .with_columns(pl.col(null_max_columns).fill_null(strategy="max"))
            .select(pl.all().fill_null(0))
            .drop("vendor_id")
            .collect()
        )
        features = features_df.select(pl.all().rank(method="min")).to_numpy()
        scores = (features * feature_weights).sum(axis=1)
        df = df.with_columns(pl.Series(name="scores", values=scores))

        # If cosine similarity is available, use it to rank vendors and then combine the ranks and sort by the min rank
        if df["cosine_similarity"].is_not_null().any():
            df = df.with_columns(
                rank=pl.col("scores").rank(method="min", descending=True),
                rank_cosine=pl.col("cosine_similarity").rank(method="min", descending=True),
            )
            df = df.with_columns(rank_min=pl.min("rank", "rank_cosine"))

            return df.sort("rank_min", descending=False, nulls_last=True)

        return df.sort("scores", descending=False, nulls_last=True)

    @staticmethod
    @newrelic.agent.function_trace()
    async def rank_vendors(features: np.ndarray) -> np.ndarray:
        runner = catboost_model.get_runner()
        result = await runner.async_run(features)

        return result

    def _shift_rank(self, rank, shift):
        return rank + ((shift - 1) * 5) + 0.1

    @newrelic.agent.function_trace()
    def sort_duplicated_chains(self, features: pl.DataFrame):
        features = features.with_columns(
            final_rank=pl.when((pl.col("chain_name_rank") >= 2) & (pl.col("chain_name").is_not_null()))
            .then(self._shift_rank(pl.col("fine_sort_scores_rank"), pl.col("chain_name_rank")))
            .otherwise(pl.col("fine_sort_scores_rank"))
        )
        features = features.sort(["final_rank"], nulls_last=True, descending=False)

        return features

    @newrelic.agent.function_trace()
    async def _fine_sort(self, features: pl.DataFrame) -> list[int]:
        model_features = features.select(
            [
                "cosine_similarity",
                "popularity_org_score",
                "new_customer_retention_14",
                "dwell_time_seconds_mean",
                "aov_eur",
                "new_order_prc",
                "promotion_order_prc",
                "menu_to_cart_rate",
                "timeofday_score",
                "rating_count",
                "delivery_time",
                "delivery_fee",
            ]
        ).to_numpy()
        fine_sort_scores = await self.rank_vendors(model_features)
        vendors = features["vendor_id"].to_numpy()

        df = features.with_columns(pl.Series("fine_sort_scores", fine_sort_scores))
        df = df.with_columns(fine_sort_scores_rank=pl.col("fine_sort_scores").rank(descending=True, method="ordinal"))
        df_pl_chain = df.unique(subset=["chain_id", "chain_name"]).sort("fine_sort_scores_rank", descending=False)
        df_pl_chain = df_pl_chain.with_columns(
            chain_name_rank=pl.col("fine_sort_scores_rank").rank(descending=False, method="ordinal").over("chain_name")
        )
        df = df.join(df_pl_chain[["chain_id", "chain_name_rank"]], on="chain_id", how="left")

        sorted_vendor_list = self.sort_duplicated_chains(df)["vendor_id"].to_list()

        self.exec_log.model_input = model_features.tolist()
        self.exec_log.model_output = dict(zip(vendors.tolist(), fine_sort_scores.tolist()))  # type: ignore

        return sorted_vendor_list

    @newrelic.agent.function_trace()
    async def prepare_features(self):
        customer_embedding = await self._get_customer_embedding(self.request.customer_id)

        # Further work can be done outside the main thread
        await self._prepare_features_from_embedding(customer_embedding)

    @ace.asyncio.to_thread
    def _prepare_features_from_embedding(self, customer_embedding: np.ndarray):
        dynamic_features = self._get_dynamic_features()
        self.stats += [("catboost.request.total_dynamic_features_cnt", dynamic_features.height)]

        dynamic_features, unavailable_vendors = split_by_vendor_availability(dynamic_features)
        self.stats += [
            ("catboost.request.avail_dynamic_features_cnt", dynamic_features.height),
            ("catboost.request.unavail_dynamic_features_pct", unavailable_vendors.height),
        ]

        self._sorted_unavailable_vendors = unavailable_vendors.sort("status").select("vendor_id")["vendor_id"].to_list()

        static_features = data.get_vendors(
            self.request.location.country_id,
            dynamic_features["vendor_id"],
            self.request.timestamp,
        ).collect()

        if customer_embedding.size > 0:
            static_features = ml.with_cosine_similarity(static_features, customer_embedding)
        else:
            self.exec_log.metadata["is_customer_embedding_found"] = False
            self.stats += [("catboost.unknown_user", 1)]
            static_features = static_features.with_columns(pl.lit(None).alias("cosine_similarity"))

        self.stats += [("catboost.request.total_static_features_cnt", static_features.height)]
        features = _combine_features(dynamic_features, static_features)

        rate_lc_to_eur = lc_to_eur(self.request.location.country_id)
        features = features.with_columns(delivery_fee=pl.col("delivery_fee") / rate_lc_to_eur)

        self._features = features

    @newrelic.agent.function_trace()
    async def sort(self) -> list[int]:
        if (top_n := self.TOP_N_FAST_SORT_LIMIT) is not None and len(self._features) > top_n:
            fast_sorted_features: "pl.DataFrame" = await self._fast_sort(self._features)
            vendors_to_fine_sort = fast_sorted_features[:top_n]
            fast_sorted_vendors = fast_sorted_features[top_n:]["vendor_id"].to_list()
        else:
            vendors_to_fine_sort = self._features
            fast_sorted_vendors = []

        fine_sorted_vendors = await self._fine_sort(vendors_to_fine_sort)
        final_sorting = fine_sorted_vendors + fast_sorted_vendors + self._sorted_unavailable_vendors
        return final_sorting
