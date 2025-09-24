"""
From https://gist.github.com/nymous/f138c7f06062b7c43c060bf03759c29e
Also see https://github.com/hynek/structlog/issues/490 for another configuration example
And
"""

import logging
import sys
from functools import cache
from typing import TYPE_CHECKING

import newrelic.api.log
import newrelic.api.time_trace
import structlog
from structlog.processors import CallsiteParameter
from structlog.types import EventDict, Processor

import ace
from ace.configs.config import LogLevel, StageType
from ace.configs.manager import ConfigManager

if TYPE_CHECKING:
    from ace.configs.config import AppLoggingConfig


def add_new_relic_context(_, __, event_dict: EventDict) -> EventDict:
    # Log entries keep the text message in the `event` field
    # (also see https://github.com/hynek/structlog/issues/35#issuecomment-591321744)
    event_dict["message"] = event_dict.pop("event")
    event_dict.update(newrelic.api.time_trace.get_linking_metadata())

    if exc_info := event_dict.get("exc_info"):
        if isinstance(exc_info, tuple):
            event_dict.update(newrelic.api.log.format_exc_info(exc_info))

    return event_dict


def drop_uvicorn_color_message(_, __, event_dict: EventDict) -> EventDict:
    """
    Uvicorn logs the message one extra time, but we don't need it
    """
    event_dict.pop("color_message", None)

    return event_dict


def drop_tracebacks(_, __, event_dict: EventDict) -> EventDict:
    """
    Just drop the traceback for now, as it can contain sensitive information (DB password)
    """
    event_dict.pop("exc_info", None)

    return event_dict


_initiated = False


@cache
def configure(debug: bool = None) -> None:
    global _initiated
    if _initiated:
        return

    if debug is None:
        debug = ace.DEBUG

    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.stdlib.ExtraAdder(),
        drop_uvicorn_color_message,
        # `timestamp` field will be replaced by FluentBit anyway (if we use this way, not New Relic's "Logs in Context")
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.CallsiteParameterAdder(
            [
                CallsiteParameter.MODULE,
                CallsiteParameter.LINENO,
                CallsiteParameter.FUNC_NAME,
            ]
        ),
    ]

    if not debug:
        shared_processors.append(add_new_relic_context)

    config = ConfigManager.load_configuration()
    log_config: "AppLoggingConfig" = config.logging
    if log_config.level is LogLevel.DEBUG:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    structlog.configure(
        processors=shared_processors
        + [
            # Prepare event dict for `ProcessorFormatter`
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(log_level),  # set the loging level
        cache_logger_on_first_use=True,
    )

    renderers = [
        drop_tracebacks,
        # structlog.processors.dict_tracebacks,
        structlog.processors.JSONRenderer(),
    ]
    if log_config.add_console_renderer:
        renderers = [structlog.dev.ConsoleRenderer(exception_formatter=structlog.dev.plain_traceback)]

    formatter = structlog.stdlib.ProcessorFormatter(
        # These run ONLY on entries that do NOT originate within structlog
        foreign_pre_chain=shared_processors,
        # These run on ALL entries after the pre_chain is done
        processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta] + renderers,
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)

    # Override third-party libraries' loggers
    # (See https://github.com/hynek/structlog/pull/115 for more info, and do not try to raise a DropEvent exception in
    # the foreign_pre_chain :)
    for logger_name in ("rocketry.scheduler", "rocketry.task"):
        logging.getLogger(logger_name).setLevel(logging.INFO)

    for logger_name in ("botocore", "aiobotocore"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    if config.stage is StageType.PROD:
        logging.getLogger("openai").setLevel(logging.WARNING)

    def handle_exception(exc_type, exc_value, exc_traceback):
        """
        Log any uncaught exception instead of letting it be printed by Python
        (but leave KeyboardInterrupt untouched to allow users to Ctrl+C to stop)
        See https://stackoverflow.com/a/16993115/3641865
        """
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        root_logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_exception
    _initiated = True
