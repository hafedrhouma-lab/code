import asyncio
from typing import TYPE_CHECKING, Sequence, Optional

import newrelic.agent
import numpy as np
import polars as pl
import pydantic
import structlog
from fastapi import HTTPException
from starlette import status

import ace
from abstract_ranking.base_logic import infer_user_embedding_two_tower
from abstract_ranking.two_tower import TTVersion
from abstract_ranking.two_tower.names import (
    CHAIN_EMBEDDINGS,
    COSINE_SIMILARITY_RANK,
    CHAIN_RANK,
    COSINE_SIMILARITY,
    CHAIN_NAME,
    CHAIN_ID,
    VENDOR_ID,
    STATUS,
    FINAL_RANK,
    ORIGINAL_RANK,
)
from ace import ml
from ace.enums import CountryShortNameUpper
from ace.model_log import LogEntry
from ace.perf import perf_manager
from ace.storage.db import assert_table_exists
from vendor_ranking import SERVICE_NAME
from vendor_ranking.base_logic import VendorBaseLogic
from vendor_ranking.input import (
    VendorFeatures,
    VendorList,
    split_by_vendor_availability,
)
from vendor_ranking.two_tower.artefacts_service import ArtefactsService, VendorArtefactsServiceBase
from vendor_ranking.two_tower.artefacts_service_registry import VendorArtefactsServiceRegistry
from vendor_ranking.two_tower.repository.users import TwoTowersUsersRepository
from vendor_ranking.two_tower.utils import (
    combine_tt_user_features,
    get_tt_dynamic_features_type,
)

if TYPE_CHECKING:
    from structlog.stdlib import BoundLogger

LOG: "BoundLogger" = structlog.get_logger()


def _shift_rank(rank, shift):
    return rank + ((shift - 1) * 5) + 0.1


def create_tt_logic_class(version: TTVersion, date: str):
    class LogicTwoTowersBase(VendorBaseLogic):
        VERSION: TTVersion = version
        NAME = f"{SERVICE_NAME}:two_towers_{version}"
        MODEL_TAG = f"{SERVICE_NAME}:two_towers_{version}_{date}"
        NICKNAME = f"two_towers_{version}"

        def __init__(
            self,
            request: VendorList,
            exec_log: LogEntry,
            artifacts_service_registry: VendorArtefactsServiceRegistry,
        ):
            super().__init__(request, exec_log)
            self.artifacts_service: ArtefactsService = artifacts_service_registry.get(
                version
            )
            self.users_repo: TwoTowersUsersRepository = TwoTowersUsersRepository(
                version=self.VERSION
            )
            self.sorted_vendors_similarity_scores: Optional[pl.DataFrame] = None
            self._unavailable_vendors: Optional[pl.DataFrame] = None
            self._vendors_with_chain_ranks: Optional[pl.DataFrame] = None
            self._customer_embedding: Optional[np.ndarray] = None
            self._dynamic_features_class = get_tt_dynamic_features_type(self.VERSION)

            if (
                request.location.country_code
                not in self.artifacts_service.activated_countries
            ):
                raise HTTPException(
                    status_code=status.HTTP_406_NOT_ACCEPTABLE,
                    detail={
                        "code": "COUNTRY_NOT_SUPPORTED_FOR_MODEL_VERSION",
                        "description": f"country not supported: {request.location.country_code}, {self.VERSION}",
                    },
                )

        @classmethod
        async def check_execution_requirements(cls, msg: str):
            """- Postgres DB must contain not empty tables with TwoTower users' and chains' features and embeddings.
            - Models must be present in S3 for all required countries and recall values.
            """
            # 1: validate database tables
            await asyncio.gather(
                *(
                    assert_table_exists(table_name=table_name, msg=msg)
                    for table_name in (
                    VendorArtefactsServiceBase.get_embeddings_table_name(cls.VERSION),
                    TwoTowersUsersRepository(
                            version=cls.VERSION
                        ).users_features_table_name,
                    )
                )
            )

            # 2: validate S3 files
            # That validation will be done in ArtefactsService.load().
            # There is no need to do here.

        @newrelic.agent.function_trace()
        async def sort(self) -> list[int]:
            assert (
                self._vendors_with_chain_ranks is not None
            ), "features are not prepared"
            assert (
                self._unavailable_vendors is not None
            ), "unavailable vendors are not prepared"

            # Further work can be done outside the main thread
            return await self._rank_and_sort()

        @ace.asyncio.to_thread
        def _rank_and_sort(self) -> list[int]:
            fine_sorted_vendor_list = self._rank_vendors(self._vendors_with_chain_ranks)
            sorted_unavailable_vendors = self._sort_unavailable_vendors()

            final_sorting = fine_sorted_vendor_list + sorted_unavailable_vendors

            return final_sorting

        @newrelic.agent.function_trace()
        def _log_request_chains_stats(self, requested_chains: Sequence[int]):
            country_code: str = self.request.location.country_code
            # Chains from the database
            chains_info = self.artifacts_service.get_embeddings(country_code)

            cnt_col_name = "count"

            # 1: Count the number of unknown chains in the request.
            known_request_chains_cnt = chains_info.filter(pl.col(CHAIN_ID).is_in(requested_chains)).select(
                pl.count().alias(cnt_col_name)
            )[cnt_col_name][0]
            requested_cnt = len(requested_chains)
            unknown_request_chains_cnt = requested_cnt - known_request_chains_cnt

            # 2: Count the number of known chains in request with absent embeddings.
            chains_with_absent_embeddings_cnt = chains_info.filter(
                pl.col(CHAIN_ID).is_in(requested_chains)
                & pl.col(CHAIN_EMBEDDINGS).is_null()
            ).select(pl.count().alias(cnt_col_name))[cnt_col_name][0]

            # 3: Count known chains with absent names.
            chains_with_absent_names_cnt = chains_info.filter(
                pl.col(CHAIN_ID).is_in(requested_chains) & pl.col(CHAIN_NAME).is_null()
            ).select(pl.count().alias(cnt_col_name))[cnt_col_name][0]

            if requested_cnt:
                avail_active_chains_pct = (known_request_chains_cnt / requested_cnt) * 100
                self.exec_log.add_metadata("avail_active_chains_pct", avail_active_chains_pct)

                avail_active_chains_without_embeddings_pct = (chains_with_absent_embeddings_cnt / requested_cnt) * 100
                self.exec_log.add_metadata(
                    "avail_active_chains_without_embeddings_pct", avail_active_chains_without_embeddings_pct
                )

                avail_active_chains_with_unknown_name_pct = (chains_with_absent_names_cnt / requested_cnt) * 100
                self.exec_log.add_metadata(
                    "avail_active_chains_with_unknown_name_pct", avail_active_chains_with_unknown_name_pct
                )

                self.stats += [
                    (
                        "2towers.request.avail_active_chains_pct",
                        avail_active_chains_pct
                    ),
                    (
                        "2towers.request.avail_active_chains_without_embeddings_pct",
                        avail_active_chains_without_embeddings_pct,
                    ),
                    (
                        "2towers.request.avail_active_chains_with_unknown_name_pct",
                        avail_active_chains_with_unknown_name_pct,
                    ),
                ]
            else:
                LOG.warning("No available vendors in request")

            self.stats += [
                ("2towers.request.avail_active_chains_without_embeddings_cnt", chains_with_absent_embeddings_cnt),
                ("2towers.request.avail_active_chains_with_unknown_name_cnt", chains_with_absent_names_cnt),
                ("2towers.request.avail_unknown_chains_cnt", unknown_request_chains_cnt),
            ]
            LOG.debug(
                f"chains_cnt: {requested_cnt}, "
                f"number of unknown chains in the request: {unknown_request_chains_cnt}, "
                f"number of known chains in request with absent embeddings: {chains_with_absent_embeddings_cnt}, "
                f"known chains with absent names: {chains_with_absent_names_cnt}"
            )

        @newrelic.agent.function_trace()
        async def prepare_features(self) -> "pl.DataFrame":
            self._customer_embedding = await self._get_customer_embedding(self.request)

            # Further work can be done outside the main thread
            return await self._prepare_vendors_with_chain_ranks()

        @ace.asyncio.to_thread
        def _prepare_vendors_with_chain_ranks(self):
            dynamic_features = self._get_dynamic_features()
            available_vendors, self._unavailable_vendors = split_by_vendor_availability(
                dynamic_features
            )
            available_vendors = available_vendors.with_columns(
                pl.arange(start=0, end=available_vendors.height).alias(ORIGINAL_RANK)
            )
            requested_chains = available_vendors[CHAIN_ID].unique(maintain_order=True)
            chains = self._get_chain_embeddings(
                self.request.location.country_code, requested_chains=requested_chains
            )

            chains_ranks = self._build_similarity_ranks(chains)
            self._vendors_with_chain_ranks = available_vendors.join(
                chains_ranks, how="left", on=CHAIN_ID
            )
            self._log_request_chains_stats(requested_chains)

            all_chains_cnt = len(dynamic_features[CHAIN_ID].unique())
            avail_chains_cnt = len(requested_chains)

            # Average of model scores for top 10 vendors (just for reporting)
            avg_cosine_similarity = chains_ranks.head(10)[COSINE_SIMILARITY].mean()

            self.stats += [
                ("2towers.request.total_chains_cnt", all_chains_cnt),
                ("2towers.request.avail_chains_cnt", avail_chains_cnt),
                (
                    "2towers.request.avail_chains_pct",
                    (avail_chains_cnt / all_chains_cnt) * 100,
                ),
                ("2towers.request.avg_cosine_similarity", avg_cosine_similarity),
            ]
            self.exec_log.add_metadata("total_chains_cnt", all_chains_cnt)
            self.exec_log.add_metadata("avail_chains_cnt", avail_chains_cnt)
            self.exec_log.add_metadata(
                "avail_chains_pct", (avail_chains_cnt / all_chains_cnt) * 100
            )

            return self._vendors_with_chain_ranks

        def _build_similarity_ranks(self, chains: "pl.DataFrame") -> "pl.DataFrame":
            chains_with_embeddings = chains.filter(
                pl.col(CHAIN_EMBEDDINGS).is_not_null()
            )
            if chains_with_embeddings.height == 0:
                return pl.DataFrame(schema=chains.schema).with_columns(
                    [
                        pl.lit(None).alias(name)
                        for name in (
                            COSINE_SIMILARITY,
                            COSINE_SIMILARITY_RANK,
                            CHAIN_RANK,
                        )
                    ]
                )

            # get cosine similarities between user and chains
            chains_with_embeddings = ml.with_cosine_similarity(
                chains_with_embeddings,
                customer=self._customer_embedding,
                embeddings_column_name=CHAIN_EMBEDDINGS,
            )

            # rank chains (but not vendors) by cosine similarity, add new column `cosine_similarity_rank`
            chains_with_embeddings = chains_with_embeddings.with_columns(
                cosine_similarity_rank=pl.col(COSINE_SIMILARITY).rank(
                    descending=True, method="ordinal"
                )
            )

            # additionally, rank chains with equal names by cosine similarity, add new column `chain_rank`
            chains_with_embeddings = chains_with_embeddings.with_columns(
                chain_rank=pl.col(COSINE_SIMILARITY_RANK)
                .rank(descending=False, method="ordinal")
                .over(CHAIN_NAME)
            )

            return chains_with_embeddings

        @newrelic.agent.function_trace()
        async def _get_customer_embedding(self, request: "VendorList") -> np.ndarray:
            customer_id, country_code = pydantic.parse_obj_as(
                tuple[int, CountryShortNameUpper],
                (request.customer_id, request.location.country_code.upper()),
            )
            static_features = await self.users_repo.get_user_static_features_two_tower(
                customer_id=customer_id, country_iso=country_code
            )

            def infer_embedding():
                features_names: set[str] = self.artifacts_service.get_features_names(
                    country_code
                )
                dynamic_features = self._dynamic_features_class.from_request(request)
                features_df, features = combine_tt_user_features(
                    dynamic_features, static_features, features_names
                )
                user_model = self.artifacts_service.get_user_model(country_code)
                with perf_manager(
                    f"infer user embedding {self.artifacts_service.get_config_repr(country_code)}",
                    attrs=self.artifacts_service.get_artifacts_attrs(country_code),
                ):
                    self.exec_log.model_input = features
                    try:
                        return infer_user_embedding_two_tower(
                            features=features_df, user_model=user_model
                        )
                    except Exception as ex:
                        recall = self.artifacts_service.default_configs[
                            country_code
                        ].recall
                        LOG.error(
                            f"Inference failed {country_code} {self.VERSION}, {recall=}, customer={customer_id}. "
                            f"Error: {ex}. "
                            f"Features:\n{features_df.to_string()}. "
                            f"Types:\n{list(zip(list(map(str, features_df.dtypes)), features_df.columns))}"
                        )
                        raise

            # Further work can be done outside the main thread
            customer_embedding = await asyncio.to_thread(infer_embedding)

            return customer_embedding

        @newrelic.agent.function_trace()
        def _get_chain_embeddings(
            self, country_code: str, requested_chains: Sequence[int]
        ) -> "pl.DataFrame":
            chains_with_embeddings = self.artifacts_service.get_embeddings(
                country_code
            )
            df = chains_with_embeddings.filter(pl.col(CHAIN_ID).is_in(requested_chains))
            return df

        @newrelic.agent.function_trace()
        def _get_dynamic_features(self) -> pl.DataFrame:
            df = pl.DataFrame(
                self.request.vendors_df, schema=VendorFeatures.df_schema, orient="row"
            )
            df = df.select([VENDOR_ID, CHAIN_ID, STATUS])
            return df

        def _sort_unavailable_vendors(self) -> list[int]:
            """
            Status:
                0 - open
                1 - closed
                2 - busy
                3 - hidden
                4 - unlisted
            Sort `status` in descending order: first `busy`, then `closed`.
            """
            return self._unavailable_vendors.sort(STATUS, descending=True)[
                VENDOR_ID
            ].to_list()

        @newrelic.agent.function_trace()
        def _sort_duplicated_chains(self, features: pl.DataFrame):
            """
            If `chain_name` is absent/unknown,
            then skip the deduplication penalty on that chain and use just `cosine_similarity_rank`.
            """
            features = features.with_columns(
                pl.when(
                    pl.col(CHAIN_RANK).is_not_null()
                    & (pl.col(CHAIN_RANK) >= 2)
                    & pl.col(CHAIN_NAME).is_not_null()
                )
                .then(_shift_rank(pl.col(COSINE_SIMILARITY_RANK), pl.col(CHAIN_RANK)))
                .otherwise(pl.col(COSINE_SIMILARITY_RANK))
                .alias(FINAL_RANK)
            )

            # additionally, sort by `original_rank` to make sorting stable
            # regarding initial vendor's ordering from the request

            features_with_no_final_rank = features.filter(
                pl.col(FINAL_RANK).is_null()
            ).sort(ORIGINAL_RANK, descending=False)

            features = features.filter(pl.col(FINAL_RANK).is_not_null()).sort(
                FINAL_RANK, ORIGINAL_RANK, descending=False
            )

            return pl.concat((features, features_with_no_final_rank))

        @newrelic.agent.function_trace()
        def _rank_vendors(self, features: pl.DataFrame) -> list[int]:
            sorted_chains = self._sort_duplicated_chains(features)
            result_vendors = sorted_chains[VENDOR_ID].to_list()
            self.sorted_vendors_similarity_scores = sorted_chains.select(
                [VENDOR_ID, COSINE_SIMILARITY]
            )
            self.exec_log.model_output = dict(
                zip(
                    features[VENDOR_ID].to_list(), features[COSINE_SIMILARITY].to_list()
                )
            )

            return result_vendors

    return LogicTwoTowersBase


LogicTwoTowersV22 = create_tt_logic_class(version=TTVersion.V22, date="20231025")
LogicTwoTowersV23 = create_tt_logic_class(version=TTVersion.V23, date="20231212")
LogicTwoTowersV3 = create_tt_logic_class(version=TTVersion.V3, date="20231212")
