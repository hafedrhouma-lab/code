import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import structlog

from abstract_ranking.two_tower import TTVersion
from abstract_ranking.two_tower.artefacts_service import ACTIVE_TT_VERSIONS
from ace.enums import CountryShortNameUpper, CountryShortName
from vendor_ranking.configs.config import get_vendor_ranking_config, VendorRankingConfig
from vendor_ranking.two_tower.artefacts_service_registry import VendorArtefactsServiceRegistry

if TYPE_CHECKING:
    from structlog.stdlib import BoundLogger

LOG: "BoundLogger" = structlog.get_logger()


@dataclass
class Context:
    artifacts_service_registry: VendorArtefactsServiceRegistry
    app_config: VendorRankingConfig
    _opened: bool = False

    @property
    def opened(self):
        return self._opened

    @classmethod
    def _prepare_countries(
        cls,
        countries,
        two_towers_countries: dict[TTVersion, set[CountryShortName]],
        config_activated_countries
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

    @classmethod
    async def instance(
        cls,
        countries: set[str] | dict[TTVersion, set[CountryShortNameUpper]] = None
    ):
        app_config: VendorRankingConfig = get_vendor_ranking_config()
        active_countries = cls._prepare_countries(
            countries,
            app_config.ranking.two_towers.countries,
            app_config.ranking.two_towers.default_countries,

        )
        artifacts_service_registry = VendorArtefactsServiceRegistry.instance(
            countries=active_countries,
            s3_app_config=app_config.storage.s3,
        )
        return cls(artifacts_service_registry=artifacts_service_registry, app_config=app_config)

    async def reload_tt_models(self):
        coros = (
            self.artifacts_service_registry.get(version).load()
            for version in ACTIVE_TT_VERSIONS
        )
        await asyncio.gather(*coros)

    async def open(self):
        # preload model artifacts
        await self.reload_tt_models()

        LOG.debug("Context opened")
        self._opened = True

    async def close(self):
        LOG.debug("Context closed")

    async def __aenter__(self):
        await self.open()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type or exc_val or exc_tb:
            # Swallow the exception and proceed with closing the context
            LOG.exception()
        await self.close()
        return self


_current_context = None


def set_context(context: Context, overwrite: bool = False):
    global _current_context
    if not overwrite:
        assert _current_context is None

    _current_context = context


def ctx() -> Optional[Context]:
    global _current_context
    return _current_context
