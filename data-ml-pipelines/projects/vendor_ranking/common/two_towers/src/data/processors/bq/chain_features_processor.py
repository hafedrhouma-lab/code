import numpy as np
import pandas as pd
from ..base_processor import BaseProcessor, process_data


class ChainFeaturesProcessor(BaseProcessor):
    def __init__(self, df: pd.DataFrame, candidate_features: list):
        super().__init__(df)
        self.candidate_features = candidate_features

    def process(self):
        chain_features = self.df
        chain_features = chain_features.drop_duplicates(subset='chain_id', keep='first')
        chain_features['tlabel'] = chain_features['tlabel'].fillna('cuisine_missing')

        chain_features = self.fill_missing_chain_features(chain_features)
        chain_features = process_data(chain_features, "negative_variables")
        chain_features = chain_features[self.candidate_features]

        return chain_features

    def fill_missing_chain_features(self, chains_df, logger=None):
        hour_pct_cols = [col for col in chains_df.columns if 'chain_hour_orders_pct' in col]
        chains_df[hour_pct_cols] = chains_df[hour_pct_cols].fillna(0)
        assert np.isclose(chains_df[hour_pct_cols].fillna(0).sum(axis=1), 1).mean() == 1

        print('Chain hour percentage distribution all sums to ~ 1')
        chains_with_nan = chains_df.isna().any(axis=1).sum()

        print(f"Number of chains with NaN values: {chains_with_nan}")

        if logger:
            logger.info(f"Number of chains with NaN values: {chains_with_nan}")

        numeric_cols = [col for col in chains_df.columns if self.is_numeric_feature(col)]
        nan_features = chains_df.isna().sum(axis=0)
        nan_features = nan_features[(nan_features > 0)].index.values
        print('features that have nan', nan_features)
        print('avg for features with nan')
        print(chains_df[nan_features].mean())

        chains_df[numeric_cols] = chains_df[numeric_cols].apply(lambda x: x.fillna(x.mean()))
        return chains_df

    @staticmethod
    def is_numeric_feature(column_name):
        if 'pct' in column_name or 'avg' in column_name \
                or 'cnt' in column_name or 'eur' in column_name \
                or 'log' in column_name:
            return True
        else:
            return False
