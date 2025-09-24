from dataclasses import dataclass
from typing import ClassVar, Type

from abstract_ranking.context import ArtefactsServiceRegistry, TTConfig
from abstract_ranking.two_tower import TTVersion
from abstract_ranking.two_tower.artefacts_service import prepare_configs, ArtefactsService
from vendor_ranking.two_tower.artefacts_service import V3VendorArtefactsService, V2VendorArtefactsService


@dataclass
class VendorArtefactsServiceRegistry(ArtefactsServiceRegistry):
    configs: ClassVar[dict[TTVersion, TTConfig]] = {
        TTVersion.V2: TTConfig(
            {"AE", "EG", "IQ", "JO", "KW", "QA", "OM", "BH"},
            prepare_configs(
                version=TTVersion.V2,
                configs=[
                    ("AE", 4858),
                    ("QA", 4705),
                ])
        ),
        TTVersion.V22: TTConfig(
            {"AE", "EG", "IQ", "JO", "KW", "QA", "OM", "BH"},
            prepare_configs(
                version=TTVersion.V22,
                configs=[
                    ("AE", 4566),
                    ("BH", 4718),
                    ("EG", 6277),
                    ("IQ", 6207),
                    ("JO", 5876),
                    ("KW", 5123),
                    ("OM", 5158),
                    ("QA", 4632),
                ])
        ),
        TTVersion.V23: TTConfig(
            {"AE", "EG", "IQ", "JO", "KW", "QA", "OM", "BH"},
            prepare_configs(
                version=TTVersion.V23,
                configs=[
                    ("BH", 4695),
                    ("EG", 6029),
                    ("IQ", 6464),
                    ("JO", 5874),
                    ("KW", 4863),
                    ("OM", 4991),
                    ("QA", 4451),
                ]
            )
        ),
        TTVersion.V3: TTConfig(
            {"AE", "EG", "IQ", "JO", "KW", "QA", "OM", "BH"},
            prepare_configs(
                version=TTVersion.V3,
                configs=[
                    ("BH", 4820),
                    ("EG", 6237),
                    ("IQ", 6480),
                    ("JO", 6071),
                    ("KW", 4926),
                    ("OM", 5169),
                    ("QA", 4508),
                ]
            )
        ),
    }

    @classmethod
    def get_artefacts_service_type(cls, version: TTVersion) -> Type[ArtefactsService]:
        match version:
            case TTVersion.V3:
                return V3VendorArtefactsService
            case TTVersion.V2 | TTVersion.V22 | TTVersion.V23:
                return V2VendorArtefactsService
            case _:
                raise ValueError(f"Model version {version} is not supported for vendors ranking")
