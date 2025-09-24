import asyncio
import logging
import operator
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, Iterable, TypeVar, Type, Generic, Callable

import asyncache
import cachetools
import newrelic.agent
import polars as pl
import structlog
import tensorflow as tf
from cachetools.keys import methodkey
from fastapi import HTTPException
from pydantic import BaseModel, Field
from starlette import status

from abstract_ranking.input import Location
from abstract_ranking.two_tower import TTVersion
from abstract_ranking.two_tower.model_artifacts import ArtefactsManager
from ace.configs.config import AppS3Config
from ace.enums import CountryShortNameUpper
from ace.perf import perf_manager

if TYPE_CHECKING:
    import pandas as pds

ArtefactsManagerBase = TypeVar("ArtefactsManagerBase", bound=ArtefactsManager)


def get_inference_path(country: str) -> str:
    base_path = os.path.join(os.getcwd(), f"inference_model_artifacts/{country}")
    os.makedirs(base_path, exist_ok=True)
    return base_path


if TYPE_CHECKING:
    import tensorflow as tf
    from structlog.stdlib import BoundLogger

LOG: "BoundLogger" = structlog.get_logger()

ACTIVE_TT_VERSIONS = (TTVersion.V22, TTVersion.V23, TTVersion.V3)


class CountryServingConfig(BaseModel):
    country: CountryShortNameUpper
    recall: int = Field(gt=0)
    version: TTVersion


def _stub_location(lat: float, lng: float, country: CountryShortNameUpper) -> Location:
    return Location(latitude=lat, longitude=lng, country_code=country, country_id=0, city_id=0, area_id=0)


def prepare_configs(
    configs: list[tuple[CountryShortNameUpper, int]], version: TTVersion
) -> dict[CountryShortNameUpper, CountryServingConfig]:
    return {
        country: CountryServingConfig(country=country, recall=recall, version=version)
        for country, recall, in configs
    }


class StaticFeatures(ABC):
    pass


S = TypeVar("S", bound="StaticFeatures")


@dataclass
class ArtefactsService(Generic[S], ABC):
    """ Utility class responsible for:
        1) Loading any king of artifacts for the specific model version
           and for all activated countries from S3 compatible storage.
        2) Loading embeddings from the Postgres database.

        For each model version (TTVersion), a separate instance of that class should be created.
        Now there are two specializations of that class: one for Vendor Ranking and another one for Items Ranking.
    """
    version: TTVersion

    # S3 connection settings
    s3_app_config: AppS3Config

    # Models' parameters for specific countries
    default_configs: dict[CountryShortNameUpper, CountryServingConfig]

    # Countries for which models must be loaded
    activated_countries: set[CountryShortNameUpper] = field(
        default_factory=lambda: {"AE", "EG", "IQ", "JO", "KW", "QA", "OM", "BH"}
    )

    _artifacts_managers: dict[CountryShortNameUpper, "ArtefactsManager"] = field(default_factory=dict)

    _embeddings: dict[CountryShortNameUpper, "pl.DataFrame"] = field(default_factory=dict)

    # instantiated models
    _user_models: dict[CountryShortNameUpper, "tf.keras.Sequential"] = field(default_factory=dict)
    _user_models_tf_func: dict[CountryShortNameUpper, Callable] = field(default_factory=dict)

    # set of features' names required by the model
    _features_names: dict[CountryShortNameUpper, set[str]] = field(default_factory=dict)

    _ttl_cache: cachetools.TTLCache = field(
        # don't call smth that more than once per 10 minutes
        default_factory=lambda: cachetools.TTLCache(maxsize=1, ttl=60 * 10)
    )

    # Stub locations for each country for warmup model inference
    _default_locations: ClassVar[dict[CountryShortNameUpper, Location]] = {
        "AE": _stub_location(lat=25.078822, lng=55.135950, country="AE"),
        "EG": _stub_location(lat=30.761942, lng=30.999939, country="EG"),
        "IQ": _stub_location(lat=33.323618, lng=44.363853, country="IQ"),
        "JO": _stub_location(lat=32.011268, lng=35.950110, country="JO"),
        "KW": _stub_location(lat=29.365390, lng=47.973167, country="KW"),
        "QA": _stub_location(lat=25.293610, lng=51.519181, country="QA"),
        "BH": _stub_location(lat=26.222815, lng=50.587439, country="BH"),
        "OM": _stub_location(lat=23.584401, lng=58.401773, country="OM"),
    }

    @property
    @abstractmethod
    def artifacts_manager_type(self) -> Type[ArtefactsManagerBase]:
        pass

    @property
    @abstractmethod
    def embeddings_table_name(self):
        pass

    @classmethod
    @abstractmethod
    def get_embeddings_table_name(cls, version: TTVersion):
        pass

    @abstractmethod
    @newrelic.agent.function_trace()
    def _make_warmup_inference(self, model: "tf.keras.Sequential", config: CountryServingConfig):
        pass

    @abstractmethod
    def assert_embedding_columns(self, columns: list[str]):
        pass

    @abstractmethod
    async def load_embeddings_from_db(self, country: str) -> "pl.DataFrame":
        pass

    @abstractmethod
    @newrelic.agent.function_trace()
    def _make_warmup_inference(self, model: "tf.keras.Sequential", config: CountryServingConfig):
        pass

    @abstractmethod
    def _get_default_user_static_features(
        self, country_code: str, customer_id: int = -1
    ) -> ArtefactsManagerBase:
        pass

    @newrelic.agent.function_trace()
    @asyncache.cachedmethod(operator.attrgetter("_ttl_cache"), key=methodkey)
    async def load(self):
        with perf_manager(
            description=f"Artifacts loaded, models instantiated: {self.version}, {self.activated_countries}",
            description_before=f"Loading artifacts: {self.version}, {self.activated_countries} ...",
            level=logging.INFO,
            attrs={"version": self.version.value},
        ):
            load_functions = [self.load_artifacts(), self.refresh_embeddings_from_db()]
            await asyncio.gather(*load_functions)
            self.make_warmup_inference()
            self._validate_loaded_data()

    def _validate_loaded_data(self):
        # validate that everything is ready
        assert self._user_models, "models are not loaded at all"
        for country in self.activated_countries:
            cfg: CountryServingConfig = self.default_configs.get(country)
            assert cfg, f"config not found for [{country}, {self.version}]: {self.default_configs}"

            err_msg = f"model is not loaded for {country}, version={self.version}, recall={cfg.recall}"
            assert self._user_models.get(country), err_msg

        assert self._embeddings, "chain embeddings are not loaded at all"
        for country in self.activated_countries:
            assert (
                df := self._embeddings.get(country)
            ) is not None, f"embeddings are not loaded for {country}"

            err_msg = f"embeddings are loaded but the dataset is empty for {country}, version={self.version}"
            assert df.height != 0, err_msg
            self.assert_embedding_columns(df.columns)

    @newrelic.agent.function_trace()
    def make_warmup_inference(self):
        for country_code, model in self._user_models.items():
            # need to measure only one specific call, and newrelic not that much useful here
            serving_config = self.default_configs.get(country_code)
            with perf_manager(
                f"Warmup inference for TwoTowers model [{serving_config}] DONE", level=logging.INFO
            ):
                self._make_warmup_inference(model=model, config=serving_config)

    def get_embeddings_per_country(self) -> (CountryShortNameUpper, Iterable["pl.DataFrame"]):
        return ((country, self._embeddings.get(country)) for country in self.activated_countries)

    @newrelic.agent.function_trace()
    async def load_artifacts(self):
        LOG.info(f"[{self.version}] Loading artifacts data for countries: {self.activated_countries}")
        coros = [
            self._load_artifacts_for_country(country)
            for country in map(lambda val: val.upper(), self.activated_countries)
        ]
        await asyncio.gather(*coros)

    async def _load_artifacts_for_country(self, country: CountryShortNameUpper):
        if not (country_config := self.default_configs.get(country)):
            LOG.warning(f"No config for country {country}. Skip it.")
            return
        base_path: str = get_inference_path(country)
        if not (artifacts_manager := self._artifacts_managers.get(country)):
            self._artifacts_managers[country] = artifacts_manager = self.artifacts_manager_type(
                base_dir=base_path,
                country=country,
                recall=country_config.recall,
                version=country_config.version,
                s3_app_config=self.s3_app_config,
            )
        reload_flags: list[bool] = await artifacts_manager.download_model_artifacts()
        user_model_files_reloaded: bool = reload_flags[0]
        if user_model_files_reloaded:
            self._user_models[country], self._features_names[country] = artifacts_manager.instantiate_user_model()
            LOG.info(f"Reloaded new user model for {country} into memory")
        else:
            LOG.info(f"Keep current user model for {country} into memory")

    def get_config_repr(self, country: CountryShortNameUpper) -> str:
        try:
            manager: "ArtefactsManager" = self._artifacts_managers[country]
            return manager.artifacts_id_repr
        except KeyError as ex:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "code": "CONFIG_NOT_FOUND_FOR_COUNTRY",
                    "description": f"config not found for country {country}",
                },
            ) from ex

    def get_artifacts_manager(self, country: CountryShortNameUpper) -> "ArtefactsManagerBase":
        try:
            return self._artifacts_managers[country]
        except KeyError as ex:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "CONFIG_NOT_FOUND_FOR_COUNTRY"},
            ) from ex

    def get_artifacts_attrs(self, country: CountryShortNameUpper) -> dict:
        return self.get_artifacts_manager(country).artifacts_id

    def get_features_names(self, country: CountryShortNameUpper) -> set[str]:
        try:
            return self._features_names[country]
        except KeyError as ex:
            err_msg = f"Query features list not found for country {country}, {self.version}"
            LOG.exception(err_msg)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "code": "QUERY_FEATURES_LIST_NOT_FOUND_FOR_COUNTRY",
                    "description": err_msg,
                },
            ) from ex

    def get_user_model(self, country: CountryShortNameUpper) -> "tf.keras.Sequential":
        try:
            return self._user_models[country]
        except KeyError as ex:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "code": "USER_MODEL_NOT_FOUND_FOR_COUNTRY",
                    "description": f"user model not found for country {country}",
                },
            ) from ex

    def get_tf_func_user_model(self, country: CountryShortNameUpper) -> "Callable":
        if model_tf_func := self._user_models_tf_func.get(country):
            return model_tf_func

        _model = self.get_user_model(country)

        def _call(model, features: "pds.DataFrame"):
            return model(features)

        model_ft_func = self._user_models_tf_func[country] = tf.function(_call)
        return model_ft_func

    def get_embeddings(self, country: CountryShortNameUpper) -> "pl.DataFrame":
        try:
            return self._embeddings[country]
        except KeyError as ex:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "code": "EMBEDDINGS_NOT_FOUND_FOR_COUNTRY",
                    "description": f"embeddings not found for country {country}",
                },
            ) from ex

    @newrelic.agent.function_trace()
    async def refresh_embeddings_from_db(self):
        coros = [
            self._refresh_embeddings_from_db(country) for country in self.activated_countries
        ]
        await asyncio.gather(*coros)

    async def _refresh_embeddings_from_db(self, country: CountryShortNameUpper) -> "pl.DataFrame":
        try:
            LOG.info(f"Loading embeddings for {country} into memory")
            new_df: "pl.DataFrame" = await self.load_embeddings_from_db(country=country)
            self._embeddings[country] = new_df
            LOG.info(
                f"Loaded {new_df.height} embeddings from DB for {country} into memory: "
                f"{new_df.estimated_size('mb'):.3f} mb"
            )
            return new_df
        except TimeoutError as ex:
            LOG.info(f"Failed to load embeddings from DB for {country} into memory. Error: {ex}")
            raise
