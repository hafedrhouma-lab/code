import os
from pathlib import Path
from typing import ClassVar, Callable, Any, Type, Iterable, TYPE_CHECKING, TypeVar, Union

import pydantic as pd
import structlog
from omegaconf import OmegaConf, DictConfig, Resolver, Container
from pydantic import BaseSettings, Field, ValidationError, BaseModel

from ace.configs.config import StageType, AppConfig

if TYPE_CHECKING:
    from structlog.stdlib import BoundLogger

LOG: "BoundLogger" = structlog.get_logger()


class EnvAppSettings(BaseSettings):
    stage: StageType = Field(env="STAGE", default=StageType.QA)


ConfigType = TypeVar("ConfigType", bound=BaseModel)


PathStr = Union[str, Path]


class ConfigManager:
    resolvers: ClassVar[dict[str, "Resolver"]] = {}

    @classmethod
    def load_configuration(
        cls,
        stage: StageType = None,
        as_dict: bool = False,
        source_paths: list[Path] = None,
        config_type: Type[ConfigType] = AppConfig,
    ) -> (ConfigType, dict):
        stage = stage or EnvAppSettings().stage
        LOG.debug(f"Loading config for {stage=}")
        source_paths = [Path(__file__).parent.resolve()] + (source_paths or [])
        configs_paths: list[Path] = []
        for source_path in source_paths:
            configs_paths.append(source_path / "base.yaml")
            if (stage_config_path := source_path / f"{stage.value}.yaml").exists():
                configs_paths.append(stage_config_path)
            else:
                LOG.warning(f"Config for {stage.value} not found in {stage_config_path.absolute()}")

        config: DictConfig = cls._load_many(configs_paths)
        return cls._convert(config, config_type=config_type, as_dict=as_dict)

    @classmethod
    def add_resolver(cls, name: str = None):
        def wrapper(func: "Callable"):
            cls.resolvers[name or func.__name__] = func
            return func

        return wrapper

    @classmethod
    def register_resolvers(cls):
        OmegaConf.clear_resolvers()
        for name, resolver in cls.resolvers.items():
            OmegaConf.register_new_resolver(name, resolver, replace=False)

    @classmethod
    def _convert(cls, config: DictConfig, config_type: Type[ConfigType], as_dict: bool = True):
        if as_dict:
            return cls._as_dict(config)
        dict_repr = cls._as_dict(config)
        try:
            return pd.parse_obj_as(config_type, dict_repr)
        except ValidationError as ex:
            raise ValueError(f"Failed to create config from value {dict_repr}") from ex

    @classmethod
    def _resolve_value(cls, cfg: "Container", key: str, throw_on_missing: bool = True) -> Any:
        return OmegaConf.select(cfg, key, throw_on_missing=throw_on_missing)

    @classmethod
    def _as_dict(cls, config: DictConfig) -> dict:
        return OmegaConf.to_container(config, resolve=True)

    @classmethod
    def _load(cls, data: PathStr) -> DictConfig:
        if not data:
            return DictConfig({})

        if isinstance(data, Path):
            return OmegaConf.load(data)
        if isinstance(data, str):
            return OmegaConf.create(data)
        raise ValueError(f"Unsupported data type {type(data)} to load config from")

    @classmethod
    def _load_many(cls, paths: Iterable[PathStr]) -> DictConfig:
        LOG.info(f"Load configs from {[str(path.relative_to(os.getcwd())) for path in paths]}")
        config = DictConfig({})
        config.merge_with(*(cls._load(path) for path in paths))
        return config


@ConfigManager.add_resolver()
def with_alias(value: Any, alias_name: str, aliases: "DictConfig"):
    setattr(aliases, alias_name, value)
    return value


def create_type_resolver(resolver_name, _class_type: Type):
    @ConfigManager.add_resolver(resolver_name)
    def resolver(value: Any) -> _class_type:
        return _class_type(value)


for class_type in (int, str, float, Path):
    create_type_resolver(f"as_{class_type.__name__.lower()}", class_type)
