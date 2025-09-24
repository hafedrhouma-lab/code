"""Class to process country search metric"""
from dataclasses import dataclass

import pandas as pd
from plotly import graph_objects as go
from src.data.datasets.fetch import FetchQueriesPerformances


@dataclass
class CountryMetric:
    """
     class to store country search metric
     and apply domain operations
    """
    date_start: str

    def __post_init__(self):
        self.data_frame = \
            FetchQueriesPerformances(
                "country_metric"
            ).get(self.__dict__)

        self.data_frame[
            "number_query_sessions_non_placed_order"
        ] = self.data_frame[
                "number_sessions"
            ] - \
            self.data_frame[
                "number_query_sessions_that_placed_order"
            ]

    def figure_search_type(self) -> go:
        """return figure object to plot"""

        df_grouped = self.group_data_frame(
            self.data_frame,
            [
                "user_selected_country",
                "search_type"
            ]
        )

        return self.stacked_bar_2d(
            df_grouped,
            "search_type",
            "item_search",
            "store_search",
            "Number of Sessions Search per Search type"
        )

    def figure_query_language(self) -> go:
        """return figure object to plot"""

        df_grouped = self.group_data_frame(
            self.data_frame,
            [
                "user_selected_country",
                "search_query_lang"
            ]
        )

        return self.stacked_bar_2d(
            df_grouped,
            "search_query_lang",
            "english",
            "arabic",
            "Number of Sessions Search per language"
        )

    @staticmethod
    def group_data_frame(
            data_frame: pd.DataFrame,
            columns_group: list) -> pd.DataFrame:
        """group by dataframe given columns"""
        return data_frame.groupby(
            columns_group
        ).sum().reset_index().sort_values(
            by="number_sessions"
        )

    @staticmethod
    def stacked_bar_2d(
            data_frame: pd.DataFrame,
            dimension,
            first_label: str,
            second_label: str,
            title: str) -> go:
        """2 dimensional stacked bar"""
        return go.Figure(
            data=[
                go.Bar(
                    name=f"{first_label} Orders",
                    x=data_frame[
                        data_frame[dimension] == first_label
                        ]["number_query_sessions_that_placed_order"],
                    y=data_frame[
                        data_frame[dimension] == first_label
                        ]["user_selected_country"],
                    offsetgroup=0,
                    orientation='h',
                ),
                go.Bar(
                    name=f"{second_label} Orders",
                    x=data_frame[
                        data_frame[dimension] == second_label
                        ]["number_query_sessions_that_placed_order"],
                    y=data_frame[
                        data_frame[dimension] == second_label
                        ]["user_selected_country"],
                    offsetgroup=1,
                    orientation='h'
                ),
                go.Bar(
                    name=f"{first_label} Non Orders",
                    x=data_frame[
                        data_frame[dimension] == first_label
                        ]["number_query_sessions_non_placed_order"],
                    y=data_frame[
                        data_frame[dimension] == first_label
                        ]["user_selected_country"],
                    offsetgroup=0,
                    base=data_frame[
                        data_frame[dimension] == first_label
                        ]["number_query_sessions_that_placed_order"],
                    orientation='h'
                ),
                go.Bar(
                    name=f"{second_label} Non Orders",
                    x=data_frame[
                        data_frame[dimension] == second_label
                        ]["number_query_sessions_non_placed_order"],
                    y=data_frame[
                        data_frame[dimension] == second_label
                        ]["user_selected_country"],
                    offsetgroup=1,
                    base=data_frame[
                        data_frame[dimension] == second_label
                        ]["number_query_sessions_that_placed_order"],
                    orientation='h'
                )
            ],
            layout=go.Layout(
                title=title,
                height=400,
                legend=dict(
                    yanchor="top",
                    y=0.3,
                    xanchor="left",
                    x=0.75
                )
            )
        )
