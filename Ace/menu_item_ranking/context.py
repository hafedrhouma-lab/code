import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Iterable, Coroutine

import newrelic.agent
import structlog

from abstract_ranking.two_tower import TTVersion
from ace.enums import CountryShortName, CountryShortNameUpper
from ace.perf import perf_manager
from menu_item_ranking.artefacts_service_registry import MenuArtefactsServiceRegistry
from menu_item_ranking.configs.config import (
    get_menu_item_ranking_config,
    MenuItemRankingConfig
)
from menu_item_ranking.user_features.registry import UserFeaturesProvidersRegistry

if TYPE_CHECKING:
    from structlog.stdlib import BoundLogger

LOG: "BoundLogger" = structlog.get_logger()


ACTIVE_MODEL_VERSIONS = (TTVersion.MENUITEM_V1,)


@dataclass
class Context:
    artifacts_service_registry: MenuArtefactsServiceRegistry
    app_config: MenuItemRankingConfig
    user_features_providers_registry: UserFeaturesProvidersRegistry = field(
        default_factory=UserFeaturesProvidersRegistry
    )

    _opened: bool = False

    @classmethod
    async def instance(
        cls,
        countries: set[str] | dict[TTVersion, set[CountryShortNameUpper]] = None
    ):
        app_config: MenuItemRankingConfig = get_menu_item_ranking_config()

        active_countries = cls._prepare_countries(
            countries,
            two_towers_countries=app_config.ranking.two_towers.countries,
            config_activated_countries=app_config.ranking.two_towers.default_countries,
        )
        artifacts_service_registry = MenuArtefactsServiceRegistry.instance(
            countries=active_countries,
            s3_app_config=app_config.storage.s3,
        )

        return cls(artifacts_service_registry=artifacts_service_registry, app_config=app_config)

    @property
    def opened(self):
        return self._opened

    @newrelic.agent.function_trace()
    async def reload_tt_user_models(self):
        with perf_manager(
            description=f"Artifacts and models was loaded for versions: {ACTIVE_MODEL_VERSIONS}",
            description_before=f"Loading artifacts and models for versions: {ACTIVE_MODEL_VERSIONS}",
            level=logging.INFO
        ):
            coros = (
                self.artifacts_service_registry.get(version).load()
                for version in ACTIVE_MODEL_VERSIONS
            )
            await asyncio.gather(*coros)

    @newrelic.agent.function_trace()
    async def reload_items_embeddings(self):
        coros: list[Coroutine] = [
            self.artifacts_service_registry.get(version).refresh_embeddings_from_db()
            for version in ACTIVE_MODEL_VERSIONS
        ]
        await asyncio.gather(*coros)

    async def open(self):
        # preload model artifacts
        await self.reload_tt_user_models()
        self.user_features_providers_registry = await UserFeaturesProvidersRegistry.instance(
            self.artifacts_service_registry, ACTIVE_MODEL_VERSIONS
        )

        LOG.info("Context is opened")
        self._opened = True

    async def close(self):
        LOG.info("Context closed")

    async def __aenter__(self):
        await self.open()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type or exc_val or exc_tb:
            # Swallow the exception and proceed with closing the context
            LOG.exception()
        await self.close()
        return self

    @classmethod
    def _prepare_countries(
        cls,
        countries,
        two_towers_countries: dict[TTVersion, set[CountryShortName]],
        config_activated_countries: Iterable[CountryShortName]
    ):
        if countries:
            if isinstance(countries, set):
                return {version: countries for version in TTVersion}
            return countries
        elif two_towers_countries:
            return two_towers_countries
        elif config_activated_countries:
            if isinstance(config_activated_countries, set):
                return {version: config_activated_countries for version in TTVersion}
            return config_activated_countries
        else:
            raise ValueError(f"Active countries are not specified")


_current_context = None


def set_context(context: Context, overwrite: bool = False):
    global _current_context
    if not overwrite:
        assert _current_context is None

    _current_context = context


def ctx() -> Optional[Context]:
    global _current_context
    return _current_context
