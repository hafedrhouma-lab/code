import abc
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Type

import structlog

from abstract_ranking.two_tower import TTVersion
from abstract_ranking.two_tower.artefacts_service import ArtefactsService, CountryServingConfig
from ace.configs.config import AppS3Config
from ace.enums import CountryShortNameUpper

if TYPE_CHECKING:
    from structlog.stdlib import BoundLogger

LOG: "BoundLogger" = structlog.get_logger()


@dataclass(kw_only=False)
class TTConfig:
    countries: set[CountryShortNameUpper]
    settings: dict[CountryShortNameUpper, CountryServingConfig]


@dataclass
class ArtefactsServiceRegistry:
    """ ArtefactsServiceRegistry
         |
        \|/
        ArtefactsService per model version
         |
        \|/
        ArtifactsManager per model version + country + recall
    """
    artifacts_services: dict[TTVersion, ArtefactsService]

    configs: ClassVar[dict[TTVersion, TTConfig]] = {}

    @classmethod
    def get_config(cls, version: TTVersion, country: "CountryShortNameUpper") -> "CountryServingConfig":
        return cls.configs[version].settings[country]

    def get(self, version: TTVersion) -> "ArtefactsService":
        return self.artifacts_services[version]

    @classmethod
    def instance(
        cls,
        # S3 connection settings
        s3_app_config: "AppS3Config",
        # overrides for active countries per model version
        countries: dict[TTVersion, set[CountryShortNameUpper]] = None
    ) -> "ArtefactsServiceRegistry":
        countries = countries or dict()
        return cls(
            artifacts_services={
                version: cls.get_artefacts_service_type(version)(
                    s3_app_config=s3_app_config,
                    activated_countries=countries.get(version) or config.countries,
                    default_configs=config.settings,
                    version=version
                )
                for version, config in cls.configs.items()
            }
        )

    @classmethod
    @abc.abstractmethod
    def get_artefacts_service_type(cls, version: TTVersion) -> Type[ArtefactsService]:
        pass
