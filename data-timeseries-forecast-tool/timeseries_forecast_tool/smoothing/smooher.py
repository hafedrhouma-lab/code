from typing import Dict, List, Tuple
from timeseries_forecast_tool.utils.log import Logger
import numpy as np
import pandas as pd
from timeseries_forecast_tool._helpers import DataFrame


class DoubleExponentialSmoothing:
    """ Class to apply a double exponential smoothing on a timeseries
    Attributes:
        span: (float) span parameters for smoothing
        beta: (float) beta parameter for smoothing
        damped: (float) damped parameter for smoothing
    """

    _timeseries_df = DataFrame(["date_sql", "y"])

    def __init__(self, smoothing_params: Dict[str, float]):
        self.span = smoothing_params.get("span")
        self.beta = smoothing_params.get("beta")
        self.damped = smoothing_params.get("damped")

        self._smoothed = None
        self._trend = None
        self._trend_smoothed = None

        self.logger = Logger(self.__class__.__name__).get_logger()

    def _holt_winters(
        self, timeseries_df: pd.DataFrame
    ) -> Tuple[List[float], List[float], List[float]]:
        self._timeseries_df = timeseries_df
        series_values = self._timeseries_df["y"].values
        n = len(self._timeseries_df)
        alpha = 2.0 / (1.0 + self.span)
        smoothed = [0.0] * n
        trend = [0.0] * n
        trend_smoothed = [0.0] * n
        avg_fst_days = 0.0

        for p in range(0, min(365, n)):
            avg_fst_days += series_values[p] / min(365, n)
        smoothed[0] = avg_fst_days
        # trend b is initialized with 0
        for i in range(1, n):
            # FIXME: if date is abnormal or event
            smoothed[i] = alpha * series_values[i] + (1 - alpha) * (
                smoothed[i - 1] + self.damped * trend[i - 1]
            )
            trend[i] = (
                self.beta * (smoothed[i] - smoothed[i - 1])
                + (1 - self.beta) * trend[i - 1]
            )
            trend_smoothed[i] = np.mean(trend[i - min(365, i) : i])
        return smoothed, trend, trend_smoothed

    def __call__(self, timeseries_df: pd.DataFrame):
        """apply double exp smoothing on given timeseries dataframe"""
        self.logger.info(f"Starting Smoothing on Timeseries")
        self._smoothed, self._trend, self._trend_smoothed = self._holt_winters(
            timeseries_df
        )

    @property
    def smoothed_timeseries_df(self) -> pd.DataFrame:
        dataframe_smoothed = {
            "date_sql": self._timeseries_df["date_sql"].values,
            "smoothed_value": self._smoothed,
        }
        return pd.DataFrame(dataframe_smoothed)

    @property
    def trend(self) -> pd.DataFrame:
        dataframe_trend = {
            "date_sql": self._timeseries_df["date_sql"].values,
            "trend_value": self._trend,
        }
        return pd.DataFrame(dataframe_trend)

    @property
    def trend_smoothed(self) -> pd.DataFrame:
        dataframe_trend_smoothed = {
            "date_sql": self._timeseries_df["date_sql"].values,
            "trend_smoothed_value": self._trend_smoothed,
        }
        return pd.DataFrame(dataframe_trend_smoothed)
