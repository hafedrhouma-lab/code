from timeseries_forecast_tool.utils.log import Logger
from timeseries_forecast_tool.interfaces.interfaces import IModelDaily
from timeseries_forecast_tool._helpers import (
    DatesIterator,
    Number,
    DataFrame,
    TimeSeriesAttributes,
)
import pandas as pd
from datetime import timedelta


class ReactiveForecast(IModelDaily):
    n_days = Number(minvalue=1, maxvalue=300)
    _timeseries_df = DataFrame(["date_sql", "y"])

    def __init__(self, n_days: int = 100):
        self.iterator_dates = None
        self._timeseries_attributes = None
        self.n_days = n_days
        self.logger = Logger(self.__class__.__name__).get_logger()

    def prepare(self, timeseries_df: pd.DataFrame):

        self.logger.info(f"Preparing to forecast for {self.n_days} days")
        self._timeseries_df = timeseries_df
        self._timeseries_attributes = TimeSeriesAttributes(self._timeseries_df)
        dates_to_forecast = DatesIterator(
            self._timeseries_attributes.last_date, self.n_days
        )
        iterator_dates = iter(dates_to_forecast)
        self.iterator_dates = iterator_dates

    def fit(self, timeseries_df: pd.DataFrame):
        pass

    def predict(self, timeseries_df: pd.DataFrame) -> pd.DataFrame:
        return self.forecast()

    def forecast(self) -> pd.DataFrame:
        """
        Forecast using rate of increase/decrease vs last year.
        Rate is computed using 7 past days vs sames days of last year.
        Forecast is last year value * rate
        Return:
            df with date and forecast associated
        """
        forecasted_values = []
        forecasted_dates = []

        self.logger.info("Starting to forecast...")
        while True:
            try:
                date_to_forecast = next(self.iterator_dates)
                date_last_year = date_to_forecast - timedelta(364)

                forecast_value = (
                    self._timeseries_attributes.increase_rate
                    * self._timeseries_attributes.value_at_date(date_last_year)
                )

                forecasted_dates.append(date_to_forecast)
                forecasted_values.append(forecast_value)
            except StopIteration:
                dataframe_forecast = {
                    "date_sql": forecasted_dates,
                    "forecasted_value": forecasted_values,
                }
                return pd.DataFrame(dataframe_forecast)
