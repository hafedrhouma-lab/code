import pandas as pd
from .interface import DataFetcherInterface
from base.v0.perf import perf_manager


class FetchData(DataFetcherInterface):
    def __init__(self, store, bq_client):
        self.store = store
        self.bq_client = bq_client
        self.fetch_strategies = {
            "sql": self._fetch_from_sql,
            "feast": self._fetch_from_feast
        }

    def fetch_data(self, source: str, description: str, **kwargs) -> pd.DataFrame:
        fetch_strategy = self.fetch_strategies.get(source)
        if not fetch_strategy:
            raise ValueError(f"Unknown data source: {source}")

        with perf_manager(f"Get {description}"):
            fetched_data = fetch_strategy(**kwargs)

        return fetched_data

    def _fetch_from_sql(self, query: str, **kwargs) -> pd.DataFrame:
        return self.bq_client.read(query)

    def _fetch_from_feast(self, entity_sql: str, features: list, **kwargs) -> pd.DataFrame:
        if not entity_sql or not features:
            raise ValueError("Feast data source requires 'entity_sql' and 'features' parameters")
        return self.store.get_historical_features(entity_df=entity_sql, features=features).to_df()
