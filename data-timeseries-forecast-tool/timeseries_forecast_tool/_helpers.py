from abc import ABC, abstractmethod
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from typing import List


class DatesIterator:
    """Class to implement an iterator on dates"""

    def __init__(self, date_start: datetime, n_days: int):
        self.date_start = date_start
        self.n_days = n_days

    def __iter__(self):
        self.n = 0
        self.date_next = self.date_start
        return self

    def __next__(self):
        if self.n < self.n_days:
            self.date_next = self.date_start + timedelta(self.n + 1)
            self.n += 1
            return self.date_next
        else:
            raise StopIteration


class Validator(ABC):
    def __set_name__(self, owner, name):
        self.private_name = "_" + name

    def __get__(self, obj, obj_type=None):
        return getattr(obj, self.private_name)

    def __set__(self, obj, value):
        self.validate(value)
        setattr(obj, self.private_name, value)

    @abstractmethod
    def validate(self, value):
        pass


class Number(Validator):
    def __init__(self, minvalue=None, maxvalue=None):
        self.minvalue = minvalue
        self.maxvalue = maxvalue

    def validate(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError(f"Expected {value!r} to be an int or float")
        if self.minvalue is not None and value < self.minvalue:
            raise ValueError(f"Expected {value!r} to be at least {self.minvalue!r}")
        if self.maxvalue is not None and value > self.maxvalue:
            raise ValueError(
                f"Expected {value!r} to be no more than {self.maxvalue!r}"
            )


class DataFrame(Validator):
    def __init__(self, expected_cols: List):
        self._expected_columns = expected_cols

    def validate(self, dataframe: pd.DataFrame):
        if list(dataframe.columns) != self._expected_columns:
            raise ValueError(
                f"Expected dataframe columns to be {self._expected_columns!r}"
            )


class EffectsNames(Validator):
    _expected_values = ["date_of_week_of_month", "holidays", "special_events"]

    def validate(self, list_values: List[str]):
        for ele in list_values:
            if ele not in self._expected_values:
                raise ValueError(
                    f"Expected parameter effects values to be in {self._expected_values!r}"
                )


class TimeSeriesAttributes:
    def __init__(self, timeseries_df: pd.DataFrame):
        self._timeseries_df = self._process(timeseries_df)

    @staticmethod
    def _process(timeseries_df: pd.DataFrame):
        timeseries_df["date_sql"] = pd.to_datetime(timeseries_df["date_sql"])
        timeseries_df["y"] = [float(e) for e in timeseries_df["y"].values]
        return timeseries_df

    @property
    def last_date(self):
        return pd.to_datetime(self._timeseries_df["date_sql"].values[-1])

    @property
    def increase_rate(self):
        values_curr_year = self._get_last_n_values(self.last_date, 6)
        values_last_year = self._get_last_n_values(
            self.last_date - timedelta(364), 6
        )
        try:
            return np.mean(
                [
                    values_curr_year[i] / values_last_year[i]
                    for i in range(len(values_last_year))
                ]
            )
        except IndexError:
            return 1.0

    def _get_last_n_values(self, date_stop: datetime, n: int) -> List:
        date_n_days_ago = date_stop - timedelta(n)

        df_series_filter = self._timeseries_df[
            (self._timeseries_df["date_sql"] >= date_n_days_ago)
            & (self._timeseries_df["date_sql"] <= date_stop)
        ]

        return list(df_series_filter["y"].values)

    def value_at_date(self, date_value: datetime) -> float:
        """return timeseries value at given date"""
        return self._timeseries_df[self._timeseries_df["date_sql"] == date_value][
            "y"
        ].values[0]
