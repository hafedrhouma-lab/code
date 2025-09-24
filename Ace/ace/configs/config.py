from enum import Enum, auto
from typing import Optional

from pydantic import BaseModel, Extra, validator


class AutoUpperName(Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name.upper()


class LogLevel(str, AutoUpperName):
    CRITICAL = auto()
    FATAL = auto()
    ERROR = auto()
    WARNING = auto()
    INFO = auto()
    DEBUG = auto()


class StageType(str, Enum):
    QA = "qa"
    PROD = "prod"
    DEV = "dev"
    TEST = "test"
    TEST_QA = "test_qa"  # run service locally, connect to QA db


class AppPostgresConfig(BaseModel):
    host: str
    port: int
    user: str
    password: str
    database: str
    connection_timeout: int
    background_query_timeout: int  # millisecond
    main_query_timeout: int  # millisecond

    def connection_info_dict(self) -> dict:
        return self.dict(exclude={"password", "background_query_timeout", "main_query_timeout"})


class AppS3Config(BaseModel):
    bucket_name: str
    server_port: Optional[int]
    server_host: Optional[str]
    access_key: Optional[str]
    secret_key: Optional[str]

    class Config:
        extra = Extra.forbid


class AppStorageConfig(BaseModel):
    s3: AppS3Config
    postgres: AppPostgresConfig


def to_bool(value: bool | str) -> bool:
    if isinstance(value, bool):
        return value

    bool_mapping = {
        "on": True,
        "off": False,
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }
    s_lower = value.lower()  # Convert the string to lowercase for case-insensitivity
    if s_lower in bool_mapping:
        return bool_mapping[s_lower]

    raise ValueError(f"Cannot convert value {value} to bool")


class AppLoggingConfig(BaseModel):
    level: LogLevel
    add_console_renderer: bool

    @validator("add_console_renderer", always=True, pre=True)
    def validate_add_console_renderer(cls, val):
        return to_bool(val)


class AppConfig(BaseModel):
    stage: StageType
    storage: AppStorageConfig
    logging: AppLoggingConfig

    class Config:
        extra = Extra.forbid
