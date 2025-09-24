from pydantic import BaseModel, AnyHttpUrl
from typing import Optional


class OptionalAssertionsConfig(BaseModel):
    headers: Optional[list[tuple[str, str]]] = None


class AssertionsConfig(BaseModel):
    status_code: int
    optional: Optional[OptionalAssertionsConfig] = None


class SetupTestConfig(BaseModel):
    url: AnyHttpUrl
    method: Optional[str] = "POST"
    assertions: AssertionsConfig


class ApiTestingConfig(BaseModel):
    setup: SetupTestConfig
