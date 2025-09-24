"""Class to process In Vendor Search queries"""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from src.data.datasets.fetch import FetchQueriesPerformances
from src.utils.log import Logger

_logger = Logger(__file__).get_logger()


@dataclass
class InVendorSearchQuery:
    """
     class to store query data and apply
     domain operations
    """
    date_start: str
    os: list
    language: list
    store_type: list
    country: str
    meta: field(default_factory=dict)

    def __post_init__(self):
        self.data_frame = \
            FetchQueriesPerformances(
                "in_vendor_search"
            ).get(self.__dict__)
        self.process_data()
        self.order_data()
        self.compute_metrics()
        self.keep_columns()

    @property
    def number_search(self) -> int:
        """number of qeried search"""
        return self.data_frame.shape[0]

    def process_data(self) -> None:
        """process dataframe columns"""
        col_names = self.data_frame.drop(
            columns="query_entered"
        ).columns

        self.data_frame[
            col_names
        ] = self.data_frame[
            col_names
        ].fillna(0).astype(int)

    def order_data(self) -> None:
        """order dataframe by most
        important columns"""
        self.data_frame = \
            self.data_frame.sort_values(
                by=['number_sessions'],
                ascending=False
            ).reset_index(drop=True)

    def compute_metrics(self) -> None:
        """compute search metrics"""

        self.data_frame["%cum_session"] = \
            np.round(
                self.data_frame[
                    "number_sessions"
                ].cumsum() / self.data_frame[
                    "number_sessions"
                ].sum() * 100,
                decimals=1
            )

        self.data_frame = self.data_frame.fillna(0)

    def keep_columns(self) -> None:
        """keep relevant columns to display"""
        cols_to_keep = [
            "query_entered",
            "number_sessions",
            "number_search",
            "%cum_session",
            "ATC_percentage",
            "CVR_percentage",
            "ZRR_percentage"
        ]

        self.data_frame = self.data_frame[cols_to_keep]

    def get_most_searched(self) -> pd.DataFrame:
        """keep most searched queries"""
        return self.data_frame[
            self.data_frame["%cum_session"] <= 80
            ].reset_index(drop=True)

    @property
    def number_filtered_search(self) -> int:
        """number search keep most searched queries"""
        return \
            self.number_search - \
            self.get_most_searched().shape[0]
