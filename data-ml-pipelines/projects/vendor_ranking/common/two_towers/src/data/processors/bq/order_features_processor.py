import pandas as pd
from ..base_processor import BaseProcessor, process_data


class OrderFeaturesProcessor(BaseProcessor):
    def __init__(self, df: pd.DataFrame, query_feature: list, test_data: bool = False):
        super().__init__(df)
        self.query_feature = query_feature
        self.test_data = test_data

    def process(self):
        order_features = self.df

        order_features['order_hour'] = order_features['order_time_utc'].dt.hour
        order_features['order_weekday'] = order_features['order_time_utc'].dt.dayofweek
        order_features['geohash6'] = order_features['geohash'].astype(str).str[:6]

        float_columns = order_features.select_dtypes(include=["float32", "float64"]).columns
        order_features[float_columns] = order_features[float_columns].fillna(0)

        order_features['freq_chains'] = order_features['freq_chains'].fillna('no_frequent_orders')
        order_features['freq_clicks'] = order_features['freq_clicks'].fillna('no_frequent_clicks')
        order_features['prev_clicks'] = order_features['prev_clicks'].fillna('no_recent_clicks')
        order_features['user_prev_chains'] = order_features['user_prev_chains'].fillna('first_order')
        order_features['prev_searches'] = order_features['prev_searches'].fillna('no_prev_search')

        cols_to_keep = self.query_feature + ['chain_id', 'account_id']
        if self.test_data:
            cols_to_keep = cols_to_keep + ['order_time_utc', 'order_date', 'geohash']

        order_features = order_features[cols_to_keep]

        order_features['sample_weight'] = 1.0
        order_features = process_data(order_features, "positive_variables")

        return order_features
