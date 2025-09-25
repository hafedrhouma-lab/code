from abc import ABC, abstractmethod
from typing import Union
from datetime import datetime, timedelta
import pandas as pd
import os
from timeseries_forecast_tool._helpers import DataFrame


KEY_DATES_PATH = "key_dates_files"


def transform_date(date_value: datetime) -> str:
    return date_value.strftime("%Y-%m-%d")


class Effects(ABC):
    """Interface for methods that deal with effects on date"""

    @abstractmethod
    def get_name(self, date_value: datetime) -> Union[str, None]:
        """
        get effect name given a date value
        Args:
            date_value (datetime): date to get effect on
        Returns:
            str: effect name at this date, if any
        """


class SeasonalEffects(Effects):
    """Class to get seasonal effect name (date of year)"""

    def __init__(self):
        pass

    def get_name(self, date_value: datetime) -> Union[str, None]:
        month = str(date_value.month)
        week_num_in_month = str(self._week_number_of_month(date_value))
        weekday = str(date_value.weekday() + 1)

        return month + "_" + week_num_in_month + "_" + weekday

    def _week_number_of_month(self, date_value: datetime) -> int:
        first_monday = self._get_first_monday_of_month(date_value)
        diff_days_first_monday_first_day = first_monday.day - 1
        diff = date_value.day - first_monday.day
        res = int(diff / 7) + 1
        if diff_days_first_monday_first_day > 3 and diff >= 0:
            res += 1
        return res

    @staticmethod
    def _get_first_monday_of_month(date_value: datetime) -> datetime:
        first_date_of_month = date_value.replace(day=1)

        for i in range(0, 10):
            tmp_dt = first_date_of_month + timedelta(days=i)
            if tmp_dt.weekday() == 0:
                return tmp_dt


class HolidayEffects(Effects):
    """Class to get holiday day name """

    _holidays_df = DataFrame(["date_sql", "holiday"])
    holidays_dates_file = os.path.join(KEY_DATES_PATH, "holiday_days.csv")

    def __init__(self):
        self._holidays_df = pd.read_csv(self.holidays_dates_file)

    def get_name(self, date_value: datetime) -> Union[str, None]:
        holiday_name_df = self._holidays_df[
            self._holidays_df["date_sql"] == transform_date(date_value)
        ]

        if len(holiday_name_df) == 0:
            return None

        return holiday_name_df["holiday"].values[0]

    def is_one(self, date_value: datetime) -> bool:
        holiday_name_df = self._holidays_df[
            self._holidays_df["date_sql"] == transform_date(date_value)
        ]

        if len(holiday_name_df) != 0:
            return True

        return False


class EventEffects(Effects):
    """Class to get event day name """

    _events_df = DataFrame(["date_sql", "event"])
    events_dates_file = os.path.join(KEY_DATES_PATH, "events_days.csv")

    def __init__(self):
        self._events_df = pd.read_csv(self.events_dates_file)

    def get_name(self, date_value: datetime) -> Union[str, None]:
        event_name_df = self._events_df[
            self._events_df["date_sql"] == transform_date(date_value)
        ]

        if len(event_name_df) == 0:
            return None

        return event_name_df["event"].values[0]

    def is_one(self, date_value: datetime) -> bool:
        event_name_df = self._events_df[
            self._events_df["date_sql"] == transform_date(date_value)
        ]

        if len(event_name_df) != 0:
            return True

        return False
