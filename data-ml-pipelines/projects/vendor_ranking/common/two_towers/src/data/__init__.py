from typing import TYPE_CHECKING, Optional
from pathlib import Path

from .datasets.query_loader import QueryLoader


import projects.vendor_ranking.common.two_towers.src.utils.bq_client as bq_client

if TYPE_CHECKING:
    from feast import FeatureStore
    from .datasets.fetch import FetchData


queries_path = Path(
    __file__).parent.resolve() / "queries/feast/"
query_loader = QueryLoader(template_dir=str(queries_path))

feature_store_relative_path = Path(
    __file__).parent.resolve() / "feature_store.yaml"


STORE: Optional["FeatureStore"] = None
FETCHER: Optional["FetchData"] = None


def get_feature_store() -> "FeatureStore":
    global STORE
    if STORE is None:
        from feast import FeatureStore
        STORE = FeatureStore(
            fs_yaml_file=feature_store_relative_path
        )
    return STORE


def get_data_fetcher() -> "FetchData":
    global FETCHER
    if FETCHER is None:
        from .datasets.fetch import FetchData
        FETCHER = FetchData(get_feature_store(), bq_client)
    return FETCHER
