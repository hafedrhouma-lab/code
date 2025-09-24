# Data Class object that contains all the data needed for drift tests
import structlog
from dataclasses import dataclass, field

import pandas as pd

from projects.vendor_ranking.two_towers_v1.model_monitoring.utils import (
    get_towers_features,
    sample_rows_by_group
)

from projects.vendor_ranking.common.two_towers.evaluation import (
    get_embeddings,
    create_embedding_df,
    rank_and_evaluate
)

LOG: "structlog.stdlib.BoundLogger" = structlog.getLogger(__name__)


@dataclass
class Dataset:
    country: str
    mlflow_model: any
    start_current_date: str
    end_current_date: str
    params: dict = field(init=False)

    # Attributes for the embeddings and performance dataframes
    user_features_training: pd.DataFrame = field(init=False)
    chain_features_training: pd.DataFrame = field(init=False)
    df_training_orders: pd.DataFrame = field(init=False)

    user_features_reference: pd.DataFrame = field(init=False)
    chain_features_reference: pd.DataFrame = field(init=False)
    performance_reference: pd.DataFrame = field(init=False)

    user_features_current: pd.DataFrame = field(init=False)
    chain_features_current: pd.DataFrame = field(init=False)
    performance_current: pd.DataFrame = field(init=False)

    def __post_init__(self):
        self.params = self.mlflow_model.two_tower_model.params
        self.start_reference_date = self.params['test_date_start']
        self.end_reference_date = self.params['test_date_end']

        LOG.info(
            f"Training Dates: "
            f"{self.params['train_date_start']} to {self.params['train_date_end']}, "
        )
        LOG.info(
            f"Reference Dates: "
            f"{self.start_reference_date} to {self.end_reference_date}, "
        )
        LOG.info(
            f"Current Dates: "
            f"{self.start_current_date} to {self.end_current_date}, "
        )

        self._prepare_training_data()
        self._prepare_reference_data()
        self._prepare_current_data()
        self._map_and_clean_ids()

    def _prepare_training_data(self):
        self.user_features_training, self.chain_features_training = get_towers_features(
            self.params['train_date_start'],
            self.params['train_date_end'],
            self.country,
            self.params.get("query_features") + ['order_date'],
            self.params.get("candidate_features"),
            training_data=True
        )
        self.user_features_training = sample_rows_by_group(
            self.user_features_training,
            'order_date',
            5000
        )
        _chain_embeddings_np_training = self.mlflow_model.two_tower_model.chain_model(
            self.chain_features_training
        ).numpy()

        self.embedding_df_chain_training = create_embedding_df(
            self.chain_features_training,
            _chain_embeddings_np_training
        )
        self.df_training_orders_temp = self._create_training_orders()

    def _prepare_reference_data(self):
        self.user_features_reference, self.chain_features_reference = get_towers_features(
            self.start_reference_date,
            self.end_reference_date,
            self.country,
            self.params.get("query_features") + ['order_date'],
            self.params.get("candidate_features")
        )
        self.user_features_reference = sample_rows_by_group(
            self.user_features_reference,
            'order_date',
            5000
        )
        _user_embeddings_reference, _chain_embeddings_np_reference = get_embeddings(
            self.mlflow_model,
            self.user_features_reference,
            self.chain_features_reference
        )
        self.embedding_df_chain_reference, self.performance_reference = rank_and_evaluate(
            _user_embeddings_reference,
            _chain_embeddings_np_reference,
            self.user_features_reference,
            self.chain_features_reference
        )
        self.performance_reference_temp = self._merge_with_embeddings(
            self.performance_reference,
            self.embedding_df_chain_reference,
            left_key='item_id',
            right_key='chain_id'
        )

    def _prepare_current_data(self):
        self.user_features_current, self.chain_features_current = get_towers_features(
            self.start_current_date,
            self.end_current_date,
            self.country,
            self.params.get("query_features") + ['order_date'],
            self.params.get("candidate_features")
        )
        self.user_features_current = sample_rows_by_group(
            self.user_features_current,
            'order_date',
            5000
        )
        _user_embeddings_current, _chain_embeddings_np_current = get_embeddings(
            self.mlflow_model,
            self.user_features_current,
            self.chain_features_current
        )
        self.embedding_df_chain_current, self.performance_current = rank_and_evaluate(
            _user_embeddings_current,
            _chain_embeddings_np_current,
            self.user_features_current,
            self.chain_features_current
        )
        self.performance_current_temp = self._merge_with_embeddings(
            self.performance_current,
            self.embedding_df_chain_current,
            left_key='item_id',
            right_key='chain_id'
        )

    def _create_training_orders(self):
        """Create and prepare item ID temp dataframe."""
        df_training_orders = self.user_features_training[
            [
                'account_id',
                'chain_id'
            ]
        ].rename(
            columns={
                'account_id': 'user_id',
                'chain_id': 'item_id'
            }
        )
        df_training_orders = self._merge_with_embeddings(
            df_training_orders,
            self.embedding_df_chain_training,
            left_key='item_id',
            right_key='chain_id'
        )
        df_training_orders['target'] = 1
        return df_training_orders

    def _map_and_clean_ids(self):
        """
        Create 1:n mapping for items and users to provide the correct format
        for the Evidently AI library and map these IDs across relevant DataFrames.
        """
        def create_mapping(series):
            """Create a 1:n mapping from a unique series."""
            return {value: idx for idx, value in enumerate(series.unique(), start=1)}

        def map_ids(df, item_mapping, user_mapping):
            """Map item and user IDs in the given DataFrame."""
            return df.assign(
                item_id=df['item_id'].map(item_mapping).dropna().astype(int),
                user_id=df['user_id'].map(user_mapping).dropna().astype(int)
            )
        item_id_mapping = create_mapping(self.performance_current_temp['item_id'])
        user_id_mapping = create_mapping(self.performance_current_temp['user_id'])

        datasets = {
            'df_training_orders': self.df_training_orders_temp,
            'performance_reference': self.performance_reference_temp,
            'performance_current': self.performance_current_temp
        }
        for attr, temp_df in datasets.items():
            setattr(
                self,
                attr,
                map_ids(
                    temp_df,
                    item_id_mapping,
                    user_id_mapping
                )
            )

    @staticmethod
    def _merge_with_embeddings(base_df, embedding_df, left_key, right_key):
        """Generic merge function to combine a dataframe with embeddings."""
        return base_df.merge(
            embedding_df,
            left_on=left_key,
            right_on=right_key,
            how='left'
        ).dropna()
