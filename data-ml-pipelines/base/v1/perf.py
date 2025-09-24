import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import structlog

if TYPE_CHECKING:
    from structlog.stdlib import BoundLogger

LOG: "BoundLogger" = structlog.get_logger()


@dataclass
class TimeInterval:
    start: float
    finish: Optional[float] = None

    @property
    def delta(self):
        if self.start and self.finish:
            return self.finish - self.start
        return None


@contextmanager
def perf_manager(
    description: str = "",
    level=logging.INFO,
    description_before: str = None,
    attrs: dict = None,
):
    attrs = attrs or {}
    interval = TimeInterval(start=time.perf_counter())
    level_name = logging.getLevelName(level).lower()
    log_func = getattr(LOG, level_name)
    if description_before:
        log_func(description_before, **attrs)
    try:
        yield interval
    finally:
        interval.finish = time.perf_counter()
        if description:
            log_func(f"{description}: {interval.delta:.5f} sec.", **attrs)
