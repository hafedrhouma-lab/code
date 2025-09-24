from dataclasses import dataclass

import tensorflow as tf

from base.v0.perf import perf_manager
from projects.vendor_ranking.common.two_towers.src.data import (
    query_loader, get_data_fetcher
)
from projects.vendor_ranking.common.two_towers.src.data.datasets.tf_data import (
    create_training_data
)
from projects.vendor_ranking.common.two_towers.src.data.processors.bq import (
    ChainFeaturesProcessor,
    OrderFeaturesProcessor
)
from projects.vendor_ranking.common.two_towers.src.model.features import get_simulated_candidate_probabilities


@dataclass
class TrainingDatasetGenerator:
    def __init__(
            self,
            train_date_start: str,
            train_date_end: str,
            params: dict
    ):
        self.train_date_start = train_date_start
        self.train_date_end = train_date_end
        self.params = params

        self.positive_data: None
        self.negative_data: None
        self.train_ds: None
        self.train_ds_prefetched: None

    def fetch_and_process_data(self):
        with perf_manager(
                description="Finished Train Data Preparation",
                description_before="Preparing Train Data for Model Training..."
        ):
            # ORDER FEATURES
            order_features_query = query_loader.load_query(
                'order_features.sql.j2',
                start_date=self.train_date_start,
                end_date=self.train_date_end,
                country_code=self.params.get("country"),
                columns=self.params.get("query_features")
            )
            df_order_features = get_data_fetcher().fetch_data(
                description='Order Features For train Data',
                source="sql",
                query=order_features_query
            )
            order_features_processor = OrderFeaturesProcessor(
                df_order_features,
                self.params.get('query_features')
            )
            order_features = order_features_processor.process()

            # CHAIN FEATURES
            chain_features_query = query_loader.load_query(
                'chain_features.sql.j2',
                start_date=self.train_date_start,
                end_date=self.train_date_end,
                country_code=self.params.get("country"),
                columns=self.params.get("candidate_features")
            )
            df_chain_features = get_data_fetcher().fetch_data(
                description='chain features',
                source="sql",
                query=chain_features_query
            )
            chain_features_processor = ChainFeaturesProcessor(
                df_chain_features,
                self.params.get('candidate_features')
            )
            chain_features = chain_features_processor.process()

            self.positive_data = order_features.merge(chain_features, on='chain_id')
            self.negative_data = chain_features

    def create_train_ds(self):
        self.train_ds = create_training_data(
            self.positive_data,
            self.negative_data,
            self.params.get('train_batch'),
            self.params.get('ns_ratio')
        )

    def simulate_and_merge_probabilities(self):
        chain_probs_df = get_simulated_candidate_probabilities(self.train_ds)
        self.positive_data = self.positive_data.merge(chain_probs_df, on='chain_id')
        self.negative_data = self.negative_data.merge(chain_probs_df, on='chain_id')

        self.positive_data['candidate_probability'] = self.positive_data['candidate_probability'].astype('float32')
        self.negative_data['candidate_probability'] = self.negative_data['candidate_probability'].astype('float32')

    def finalize_train_ds(self):
        self.train_ds = create_training_data(
            self.positive_data,
            self.negative_data,
            self.params.get('train_batch'),
            self.params.get('ns_ratio')
        )

        self.train_ds_prefetched = self.train_ds.prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE
        )

    def generate(self):
        self.fetch_and_process_data()
        self.create_train_ds()
        self.simulate_and_merge_probabilities()
        self.finalize_train_ds()

        return self.train_ds_prefetched
