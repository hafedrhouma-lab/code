import importlib
from datetime import timedelta

import numpy as np
import pandas as pd
from typing import Dict, List
from timeseries_forecast_tool.utils.log import Logger
from timeseries_forecast_tool.interfaces.interfaces import IModelDaily
from timeseries_forecast_tool._helpers import (
    DatesIterator,
    Number,
    DataFrame,
    EffectsNames,
    TimeSeriesAttributes,
)
from timeseries_forecast_tool.smoothing import DoubleExponentialSmoothing
from timeseries_forecast_tool.anomalies import Anomalies, is_outlier

EFFECTS = importlib.import_module(
    "timeseries_forecast_tool.calendar_forecast.effects"
)

MEAN_SMOOTHING_PARAM = 0.3


def smooth_effects(x, param):
    ans = np.mean(x)
    for i in range(1, len(x)):
        ans = param * x[i] + (1.0 - param) * ans
    return ans


def compute_smoothed_effects(
    dict_effects: Dict[str, List[float]]
) -> Dict[str, float]:
    """compute smoothed mean of calendar_effects and holiday_days_effects
    Args: dict_effects: dictionary of {effects_name: list of values to be smoothed}
    Return:
        Dictionary of {effect_name: smoothed effect}
    """
    dict_effects_smoothed = {}
    for key, values in dict_effects.items():
        dict_effects_smoothed[key] = smooth_effects(
            dict_effects[key], MEAN_SMOOTHING_PARAM
        )

    return dict_effects_smoothed


def compute_effects(
    smoothed_values_df: pd.DataFrame, timeseries_df: pd.DataFrame
) -> pd.DataFrame:
    """Compute rate of increase/decrease per date for given timeseries
    and associate smoothed values
    Args:
        smoothed_values_df: smoothed timeseries used to compute effects
        timeseries_df: the timeseries itself to be computed effects on
    Return:
        pd.Dataframe: Ex: 2021-01-01: 1.2, 2021-01-02: 0.4 etc
    """
    effect_values = map(
        lambda x, y: x / y,
        timeseries_df["y"].values,
        smoothed_values_df["smoothed_value"].values,
    )
    dataframe_effects = {
        "date_sql": timeseries_df["date_sql"].values,
        "effect_value": effect_values,
    }
    return pd.DataFrame(dataframe_effects)


class CalendarForecast(IModelDaily):
    EFFECT_TYPE = {
        "holidays": {"cls": "HolidayEffects"},
        "special_events": {"cls": "EventEffects"},
    }

    n_days = Number(minvalue=1, maxvalue=300)
    timeseries_df = DataFrame(["date_sql", "y"])
    effects_names = EffectsNames()

    def __init__(
        self,
        smoothing_params: Dict[str, float],
        n_days: int,
        effects_names: List[str] = None,
    ):
        self.dict_effects = None
        self.smoother = None
        if effects_names is None:
            self.effects_names = ["date_of_week_of_month"]
        else:
            self.effects_names = effects_names
        self.smoothing_params = smoothing_params
        self.iterator_dates = None
        self._timeseries_attributes = None
        self._outliers = []
        self.n_days = n_days
        self.logger = Logger(self.__class__.__name__).get_logger()

    def prepare(self, timeseries_df: pd.DataFrame):
        self.logger.info(f"Preparing to forecast for {self.n_days} days")
        self.timeseries_df = timeseries_df
        self._timeseries_attributes = TimeSeriesAttributes(self.timeseries_df)
        dates_to_forecast = DatesIterator(
            self._timeseries_attributes.last_date, self.n_days
        )
        iterator_dates = iter(dates_to_forecast)
        self.iterator_dates = iterator_dates

    def fit(self, timeseries_df: pd.DataFrame):
        smoother = DoubleExponentialSmoothing(self.smoothing_params)
        smoother(timeseries_df)

        self.smoother = smoother
        dataframe_effects = compute_effects(
            smoother.smoothed_timeseries_df, timeseries_df
        )

        anomalies_dates = Anomalies().anomalies_dates

        dict_effects = {}
        for key in self.effects_names:
            dict_effects[key] = {}
        dict_effects["date_of_week_of_month"] = {}


        for index, row in dataframe_effects.iterrows():
            if row["date_sql"] in anomalies_dates:
                continue

            effect_type = "date_of_week_of_month"
            effect_name = getattr(EFFECTS, "SeasonalEffects")().get_name(
                row["date_sql"]
            )

            if self.effects_names != ["date_of_week_of_month"]:
                effects = {
                    effect: getattr(
                        EFFECTS, self.EFFECT_TYPE.get(effect)["cls"]
                    )().is_one(row["date_sql"])
                    for effect in self.effects_names
                }

                if sum(effects.values()) > 1:
                    continue

                if sum(effects.values()) == 1:
                    effect_type = list(
                        dict(filter(lambda elem: elem[1] is True, effects.items()))
                    )[0]

                    effect_name = getattr(
                        EFFECTS, self.EFFECT_TYPE.get(effect_type)["cls"]
                    )().get_name(row["date_sql"])
                else:
                    if is_outlier(row["effect_value"]):
                        self._outliers.append(row["date_sql"])
                        continue

            if effect_name not in dict_effects[effect_type].keys():
                dict_effects[effect_type][effect_name] = []

            dict_effects[effect_type][effect_name].append(row["effect_value"])

        for effect_type, effects_dict in dict_effects.items():
            dict_effects[effect_type] = compute_smoothed_effects(effects_dict)
        self.dict_effects = dict_effects

    def predict(self, timeseries_df: pd.DataFrame) -> pd.DataFrame:
        return self.forecast()

    def forecast(self) -> pd.DataFrame:
        level_line_dataframe = self._compute_forecast_line()
        forecasted_values = []
        forecasted_dates = []

        if "special_events" in self.effects_names:
            special_events_dataframe = pd.DataFrame(
                list(
                    zip(level_line_dataframe["date_sql"].values, [""] * self.n_days)
                ),
                columns=["date_sql", "event_name"],
            )

        self.logger.info(
            f"Starting to forecast from {level_line_dataframe['date_sql'].values[0]} "
            f"to {level_line_dataframe['date_sql'].values[-1]}"
        )
        while True:
            try:
                date_to_forecast = next(self.iterator_dates)

                effect_name = getattr(EFFECTS, "SeasonalEffects")().get_name(
                    date_to_forecast
                )

                forecast_value = (
                    self.dict_effects["date_of_week_of_month"].get(effect_name, 1)
                    * level_line_dataframe[
                        level_line_dataframe.date_sql == date_to_forecast
                    ]["level_value"].values[0]
                )

                if self.effects_names != ["date_of_week_of_month"]:
                    effects = {
                        effect: getattr(
                            EFFECTS, self.EFFECT_TYPE.get(effect)["cls"]
                        )().get_name(date_to_forecast)
                        for effect in self.effects_names
                    }

                    effect_type = list(
                        dict(
                            filter(lambda elem: elem[1] is not None, effects.items())
                        )
                    )

                    if effect_type:
                        forecast_value = (
                            np.prod(
                                [
                                    self.dict_effects[e].get(effects[e], 1)
                                    for e in effect_type
                                ]
                            )
                            * level_line_dataframe[
                                level_line_dataframe.date_sql == date_to_forecast
                            ]["level_value"].values[0]
                        )
                        if "special_events" in effect_type:
                            special_events_dataframe.loc[
                                special_events_dataframe["date_sql"]
                                == date_to_forecast,
                                "event_name",
                            ] = effects["special_events"]

                forecasted_dates.append(date_to_forecast)
                forecasted_values.append(forecast_value)

            except StopIteration:
                dataframe_forecast = {
                    "date_sql": forecasted_dates,
                    "forecasted_value": forecasted_values,
                }
                forecast_dataframe = pd.DataFrame(dataframe_forecast)
                if "special_events" in self.effects_names:
                    forecast_dataframe = forecast_dataframe.merge(
                        special_events_dataframe, how="left", on=["date_sql"]
                    )
                return forecast_dataframe

    def _compute_forecast_line(self):
        s = (
            list(self.smoother.smoothed_timeseries_df["smoothed_value"].values)
            + [0.0] * self.n_days
        )
        trend = list(self.smoother.trend["trend_value"].values) + [0.0] * self.n_days
        trend_smoothed = (
            list(self.smoother.trend_smoothed["trend_smoothed_value"].values)
            + [0.0] * self.n_days
        )

        n = len(self.timeseries_df)

        for i in range(n, n + self.n_days):
            s[i] = max(
                0.0,
                s[i - 1] + self.smoothing_params["damped"] * trend_smoothed[i - 1],
            )
            trend[i] = (
                self.smoothing_params["beta"] * (s[i] - s[i - 1])
                + (1 - self.smoothing_params["beta"]) * trend[i - 1]
            )
            trend_smoothed[i] = np.mean(trend[i - min(365, i) : i])

        dates_level_line = pd.date_range(
            start=self._timeseries_attributes.last_date + timedelta(1),
            periods=self.n_days,
        )
        level_line = s[n : n + self.n_days]

        df = {"date_sql": dates_level_line, "level_value": level_line}

        self.logger.info(f"Forecasting line has been computed.")
        return pd.DataFrame(df)
