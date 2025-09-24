from decouple import config

from . import api_server
from . import log, asyncio
from .service import AceService

STAGE = config("STAGE", default="qa")
DEBUG = not config("NEW_RELIC_APP_NAME", default="")

__all__ = [
    "api_server",
    "log",
    "asyncio",
    "AceService",
]
