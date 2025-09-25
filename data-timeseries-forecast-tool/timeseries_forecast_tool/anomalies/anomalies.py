"""Definiton of classes for anomalies/non quality/outliers data"""
from dataclasses import dataclass
import numpy as np
import os
import pandas as pd
import yaml
from box import Box
from datetime import date
from timeseries_forecast_tool.utils.log import Logger

_logger = Logger("Anomalies").get_logger()


def is_outlier(
        effect_value: float,
        min_threshold: float = 0.01,
        max_threshold: float = 4.0
) -> bool:
    """
    check if value is outlier
    :param
    effect_value: a given value
    (will be mostly effect=smoothed / timeseries computed by
    calendar forecast model)"""

    return np.logical_or(
        effect_value < min_threshold,
        effect_value > max_threshold
    )


@dataclass
class Anomalies:
    """Class to return Anomalies dates"""

    KEY_DATES_PATH = "key_dates_files"
    ANOMALIES_FILE_PATH = os.path.join(KEY_DATES_PATH, "anomalies.yml")

    """
    Example of anomalies file:
    anomalies:
        covid_19:
            descritpion: 
                period of covid19 pandemy and lockdowns
            type:
                range
            date:
                start: 2021 - 03 - 01
                end: 2021 - 10 - 01
        other:
            descritpion: fake event
            type: unique
            date: 2022 - 01 - 01
    """

    def __post_init__(self):
        factory = {
            "range": self.range_handler,
            "unique": self.unique_handler
        }

        self._anomalies = []

        try:
            with open(self.ANOMALIES_FILE_PATH) as yml_file:
                anomalies_box = Box(yaml.safe_load(yml_file))

            events_dict = anomalies_box.anomalies
            events_list = events_dict.keys()

            for event in events_list:
                curr_event = events_dict.get(event)

                handler = factory.get(curr_event.get('type'))

                self._anomalies.extend(handler(curr_event))

        except FileNotFoundError:
            _logger.info(f"File {self.ANOMALIES_FILE_PATH} should exist")

    @property
    def anomalies_dates(self):
        return self._anomalies

    @staticmethod
    def range_handler(event: dict):
        dates_range = pd.date_range(
            event.get("date").get("start"),
            event.get("date").get("end"),
            freq='D',
        )
        return [
            date(e.year, e.month, e.day) for e in dates_range
        ]

    @staticmethod
    def unique_handler(event):
        return [event.get("date")]
