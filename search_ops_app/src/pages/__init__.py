"""load necessary info to run app"""
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from box import Box
import pandas as pd
import yaml
from src.utils.log import Logger

_logger = Logger("Resources").get_logger()

__here__ = os.path.dirname(os.path.abspath(__file__))


DATE_START = (
        datetime.today() - timedelta(180)
).strftime("%Y-%m-%d")


@dataclass
class UserSpecifications:
    """parameters for app"""
    USER_SPECIFICATION_FILE = os.path.join(
        __here__,
        "resources/users_specifications.yaml"
    )

    def __post_init__(self):
        try:
            with open(self.USER_SPECIFICATION_FILE) as yml_file:
                self.specifications = Box(yaml.safe_load(yml_file))
        except FileNotFoundError:
            _logger.info(
                f"File {self.USER_SPECIFICATION_FILE} should exist"
            )

    @property
    def os_values(self) -> list:
        """possible os values"""
        return list(self.specifications.os.value)

    @property
    def language_values(self) -> list:
        """possible language values"""
        return list(self.specifications.language.value)

    @property
    def search_type_values(self) -> list:
        """possible search type values"""
        return list(self.specifications.search_type.value)

    @property
    def store_type_values(self) -> list:
        """possible store type (sub_vertical) values"""
        return list(self.specifications.store_type.value)


@dataclass
class GeographyMapping:
    """all possible value of user selected geography
    (queried from fct_sub_session_search)
    used in homepage_search
    """
    GEOGRAPHY_MAPPING_FILE: str

    def __post_init__(self):
        try:
            self.df_geography = pd.read_csv(
                self.GEOGRAPHY_MAPPING_FILE
            )
        except FileNotFoundError:
            _logger.info(
                f"File {self.GEOGRAPHY_MAPPING_FILE} should exist"
            )

    def get_countries(self) -> list:
        """all countries where talabat grocery exists"""
        return list(
                   set(
                       self.df_geography["user_selected_country"].values
                   )
               )

    def get_cities(self, country: str) -> list:
        """all cities where talabat grocery exists"""
        return list(
                   set(
                       self.df_geography.query(
                           "user_selected_country == @country"
                       )["user_selected_city"].values
                   )
               )

    def get_areas(self, country: str, city: str) -> list:
        """all areas where talabat grocery exists"""
        return list(
                   set(
                       self.df_geography.query(
                           "user_selected_country == @country &"
                           "user_selected_city == @city"
                       )["area_name"].values
                   )
               )

    def get_stores(self, country: str) -> list:
        """all chain stores names in selected location"""
        return list(
                   set(
                       self.df_geography.query(
                           "user_selected_country == @country"
                       )["chain_name_en"].values
                   )
               )