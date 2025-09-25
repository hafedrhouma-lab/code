# utils/logger.py
from __future__ import annotations

import logging


def get_logger(name: str, level: int | str = "INFO") -> logging.Logger:
    """
    Create/retrieve a module-level logger with a single StreamHandler.
    Safe to call multiple times; won't add duplicate handlers.
    """
    logger = logging.getLogger(name)

    # Normalize level string -> int
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    logger.setLevel(level)

    # Add one console handler if none present
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
        datefmt = "%Y-%m-%d %H:%M:%S"
        handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        logger.addHandler(handler)
        logger.propagate = False

    return logger
