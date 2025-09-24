import contextlib

from typing import (

    Iterator,
    List,
    Literal,
    Optional,

    Union,
)

import pandas as pd


from feast.feature_view import DUMMY_ENTITY_ID, DUMMY_ENTITY_VAL, FeatureView
from feast.infra.offline_stores import offline_utils
from feast.infra.offline_stores.offline_store import (
    OfflineStore,
    RetrievalJob,
    RetrievalMetadata,
)
from feast.infra.registry.base_registry import BaseRegistry
from feast.on_demand_feature_view import OnDemandFeatureView
from feast.repo_config import FeastConfigBaseModel, RepoConfig


# from .bigquery_source import (
#     BigQueryLoggingDestination,
#     BigQuerySource,
#     SavedDatasetBigQueryStorage,
# )

try:
    from google.api_core import client_info as http_client_info
    from google.api_core.exceptions import NotFound
    from google.auth.exceptions import DefaultCredentialsError
    from google.cloud import bigquery
    from google.cloud.bigquery import Client, SchemaField, Table
    from google.cloud.storage import Client as StorageClient

except ImportError as e:
    from feast.errors import FeastExtrasDependencyImportError

    raise FeastExtrasDependencyImportError("gcp", str(e))

try:
    from google.cloud.bigquery._pyarrow_helpers import _ARROW_SCALAR_IDS_TO_BQ
except ImportError:
    try:
        from google.cloud.bigquery._pandas_helpers import (  # type: ignore
            ARROW_SCALAR_IDS_TO_BQ as _ARROW_SCALAR_IDS_TO_BQ,
        )
    except ImportError as e:
        raise FeastExtrasDependencyImportError("gcp", str(e))

from feast.infra.offline_stores.bigquery import BigQueryOfflineStore
from feast.infra.offline_stores.bigquery import BigQueryOfflineStoreConfig
from feast.infra.offline_stores.bigquery import BigQueryRetrievalJob
from feast.infra.offline_stores.bigquery_source import (
    BigQuerySource,
)
from feast.infra.offline_stores.bigquery import _get_bigquery_client
from feast.infra.offline_stores.bigquery import _get_table_reference_for_new_entity
from feast.infra.offline_stores.bigquery import _get_entity_schema
from feast.infra.offline_stores.bigquery import _get_entity_df_event_timestamp_range
from feast.infra.offline_stores.bigquery import _upload_entity_df
from feast.infra.offline_stores.bigquery import block_until_done
from feast.infra.offline_stores.bigquery import MULTIPLE_FEATURE_VIEW_POINT_IN_TIME_JOIN



class BigQueryCustomOfflineStoreConfig(BigQueryOfflineStoreConfig):
    type: Literal[
        "projects.vendor_ranking.hf_transformers.utils.bigquery_custom_offline_store.BigQueryCustomOfflineStore"
    ] = "projects.vendor_ranking.hf_transformers.utils.bigquery_custom_offline_store.BigQueryCustomOfflineStore"


class BigQueryCustomOfflineStore(BigQueryOfflineStore):

    @staticmethod

    def get_historical_features(
        config: RepoConfig,
        feature_views: List[FeatureView],
        feature_refs: List[str],
        entity_df: Union[pd.DataFrame, str],
        registry: BaseRegistry,
        project: str,
        full_feature_names: bool = False,
    ) -> RetrievalJob:
        # TODO: Add entity_df validation in order to fail before interacting with BigQuery
        assert isinstance(config.offline_store, BigQueryCustomOfflineStoreConfig)
        for fv in feature_views:
            assert isinstance(fv.batch_source, BigQuerySource)
        project_id = (
            config.offline_store.billing_project_id or config.offline_store.project_id
        )
        client = _get_bigquery_client(
            project=project_id,
            location=config.offline_store.location,
        )

        assert isinstance(config.offline_store, BigQueryCustomOfflineStoreConfig)
        if config.offline_store.billing_project_id:
            dataset_project = str(config.offline_store.project_id)
        else:
            dataset_project = client.project
        table_reference = _get_table_reference_for_new_entity(
            client,
            dataset_project,
            config.offline_store.dataset,
            config.offline_store.location,
        )

        entity_schema = _get_entity_schema(
            client=client,
            entity_df=entity_df,
        )

        entity_df_event_timestamp_col = (
            offline_utils.infer_event_timestamp_from_entity_df(entity_schema)
        )

        entity_df_event_timestamp_range = _get_entity_df_event_timestamp_range(
            entity_df,
            entity_df_event_timestamp_col,
            client,
        )

        @contextlib.contextmanager
        def query_generator() -> Iterator[str]:
            _upload_entity_df(
                client=client,
                table_name=table_reference,
                entity_df=entity_df,
            )

            expected_join_keys = offline_utils.get_expected_join_keys(
                project, feature_views, registry
            )

            offline_utils.assert_expected_columns_in_entity_df(
                entity_schema, expected_join_keys, entity_df_event_timestamp_col
            )

            # Build a query context containing all information required to template the BigQuery SQL query
            query_context = offline_utils.get_feature_view_query_context(
                feature_refs,
                feature_views,
                registry,
                project,
                entity_df_event_timestamp_range,
            )

            # Generate the BigQuery SQL query from the query context
            query = offline_utils.build_point_in_time_query(
                query_context,
                left_table_query_string=table_reference,
                entity_df_event_timestamp_col=entity_df_event_timestamp_col,
                entity_df_columns=entity_schema.keys(),
                query_template=MULTIPLE_FEATURE_VIEW_POINT_IN_TIME_JOIN,
                full_feature_names=full_feature_names,
            )

            try:
                yield query
            finally:
                # Asynchronously clean up the uploaded Bigquery table, which will expire
                # if cleanup fails
                client.delete_table(table=table_reference, not_found_ok=True)

        return BigQueryCustomRetrievalJob(
            query=query_generator,
            client=client,
            config=config,
            full_feature_names=full_feature_names,
            on_demand_feature_views=OnDemandFeatureView.get_requested_odfvs(
                feature_refs, project, registry
            ),
            metadata=RetrievalMetadata(
                features=feature_refs,
                keys=list(entity_schema.keys() - {entity_df_event_timestamp_col}),
                min_event_timestamp=entity_df_event_timestamp_range[0],
                max_event_timestamp=entity_df_event_timestamp_range[1],
            ),
        )

class BigQueryCustomRetrievalJob(BigQueryRetrievalJob):
    def _execute_query(
            self, query, job_config=None, timeout: Optional[int] = None
    ) -> Optional[bigquery.job.query.QueryJob]:
        bq_job = self.client.query(query, job_config=job_config)

        if job_config and job_config.dry_run:
            print(
                "This query will process {} bytes.".format(bq_job.total_bytes_processed)
            )
            return None

        block_until_done(client=self.client, bq_job=bq_job, timeout=3600*3)
        return bq_job



