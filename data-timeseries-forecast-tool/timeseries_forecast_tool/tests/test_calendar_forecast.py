import pytest
import random
from datetime import datetime, timedelta
import pandas as pd
from timeseries_forecast_tool.calendar_forecast.effects import (
    SeasonalEffects,
    HolidayEffects,
)
from timeseries_forecast_tool.smoothing import DoubleExponentialSmoothing
from timeseries_forecast_tool.calendar_forecast import CalendarForecast


@pytest.fixture
def smoothing_params():
    """smoothing params fixture for smoother"""
    return {'span': 500, 'beta': 0.0005, 'damped': 1}


@pytest.fixture()
def test_dataframe():
    """test dataframe fixture for testing purposes"""
    start_date = datetime(2017, 1, 1)
    n_days = 1500
    dates_values = [
        (start_date + timedelta(days=i)).isoformat() for i in range(n_days)
    ]  # date range from 2017-01-01 to 2021-02-08
    y_values = random.sample(range(80000, 120000), n_days)
    return pd.DataFrame(list(zip(dates_values, y_values)), columns=['date_sql', 'y'])


@pytest.fixture()
def holidays_dataframe():
    """holiday dataframe fixture for testing purposes"""
    holidays_dates = [datetime(2021, 11, 25), datetime(2021, 12, 25)]
    holidays_values = ["thanksgiving", "christmas"]
    return pd.DataFrame(
        list(zip(holidays_dates, holidays_values)), columns=['date_sql', 'holiday']
    )


def test_seasonal_effects():
    seasonal_effects = SeasonalEffects()
    effect_name = seasonal_effects.get_name(datetime(2022, 2, 28))
    assert effect_name == '2_5_1'


def test_smoother(test_dataframe, smoothing_params):
    smoother = DoubleExponentialSmoothing(smoothing_params)
    smoother(test_dataframe)
    assert len(smoother.smoothed_timeseries_df) == len(test_dataframe)


def test_compute_forecast_line(test_dataframe, smoothing_params):
    n_days = 100
    calendar_forecast = CalendarForecast(
        smoothing_params, n_days, ["date_of_week_of_month"]
    )
    calendar_forecast.prepare(test_dataframe)
    calendar_forecast.fit(test_dataframe)
    level_line_dataframe = calendar_forecast._compute_forecast_line()
    assert level_line_dataframe["date_sql"][0] == datetime(2021, 2, 9)
    assert len(level_line_dataframe) == n_days
    assert level_line_dataframe["date_sql"][0] == next(
        calendar_forecast.iterator_dates
    )


def test_calendar_effect_not_learned(smoothing_params):
    start_date = datetime(2022, 2, 1)
    n_days = 2
    dates_values = [
        (start_date + timedelta(days=i)).isoformat() for i in range(n_days)
    ]
    y_values = random.sample(range(80000, 120000), n_days)

    test_dataframe = pd.DataFrame(
        list(zip(dates_values, y_values)), columns=['date_sql', 'y']
    )
    n_days_to_forecast = 1
    calendar_forecast = CalendarForecast(smoothing_params, n_days_to_forecast)
    calendar_forecast.prepare(test_dataframe)
    calendar_forecast.fit(test_dataframe)

    forecast_dataframe = calendar_forecast.predict(test_dataframe)
    level_line_dataframe = calendar_forecast._compute_forecast_line()

    assert list(forecast_dataframe.values[0]) == list(level_line_dataframe.values[0])
