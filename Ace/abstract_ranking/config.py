from itertools import chain
from typing import Iterable, Optional

from pydantic import BaseModel, validator, Field, constr

from abstract_ranking.two_tower import TTVersion
from ace.enums import CountryShortName


class LogicsConfig(BaseModel):
    # Maps logic's nickname to countries, where that logic must be used as default
    countries_logic: Optional[dict[str, set[CountryShortName]]] = Field(default_factory=dict)

    # Default logic's nickname for any countries
    # except the once specified in `countries_logic`
    default: str

    @validator("countries_logic", pre=True, always=True)
    def validate_countries_logic(cls, val: dict[str, Iterable[CountryShortName]]):
        return {key: set(chain(map(str.upper, values), map(str.lower, values))) for key, values in val.items()}

    def check_correctness(self):
        all_countries = set()
        for countries in self.countries_logic.values():
            if intersection := all_countries.intersection(countries):
                raise ValueError(f"Countries {intersection} have conflicting default algorithms")
            all_countries |= countries


class ExpLogicsConfig(LogicsConfig):
    experiment_name: constr(strip_whitespace=True, min_length=1)


class RankingConfig(BaseModel):
    default: LogicsConfig
    control: LogicsConfig
    holdout: list[ExpLogicsConfig] = Field(default_factory=list)
    available_countries: dict[str, CountryShortName]

    def check_correctness(self):
        self.default.check_correctness()
        self.control.check_correctness()


class TwoTowersServingConfig(BaseModel):
    default_countries: set[CountryShortName]
    countries: dict[TTVersion, set[CountryShortName]]

    @validator("default_countries", pre=True, always=True)
    def validator_default_countries(cls, val):
        return set(map(str.upper, val))

    @validator("countries", pre=True, always=True)
    def validate_countries(cls, val):
        return {
            version: set(map(str.upper, countries))
            for version, countries in val.items()
        }
