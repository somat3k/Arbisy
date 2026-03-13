"""
Logging utilities.

Provides a pre-configured logger with:
- Colored console output (via colorlog)
- Optional rotating file handler
- Consistent format across all modules
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional

try:
    import colorlog
    _HAS_COLORLOG = True
except ImportError:
    _HAS_COLORLOG = False

_LOGGERS: dict[str, logging.Logger] = {}

LOG_FORMAT = "%(asctime)s [%(levelname)-8s] %(name)s — %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

COLOR_FORMAT = (
    "%(log_color)s%(asctime)s [%(levelname)-8s]%(reset)s "
    "%(cyan)s%(name)s%(reset)s — %(message)s"
)
LOG_COLORS = {
    "DEBUG":    "white",
    "INFO":     "green",
    "WARNING":  "yellow",
    "ERROR":    "red",
    "CRITICAL": "bold_red",
}


def get_logger(
    name: str,
    level: Optional[str] = None,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Return a named logger.  Subsequent calls with the same `name` return the
    cached instance.

    Parameters
    ----------
    name:     Logger name (usually ``__name__`` of the calling module).
    level:    Override log level (default: reads LOG_LEVEL env var or INFO).
    log_file: Path for a rotating file handler (default: reads LOG_FILE env var).
    """
    if name in _LOGGERS:
        return _LOGGERS[name]

    resolved_level = level or os.getenv("LOG_LEVEL", "INFO")
    numeric_level = getattr(logging, resolved_level.upper(), logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)
    logger.propagate = False

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(numeric_level)
    if _HAS_COLORLOG:
        formatter = colorlog.ColoredFormatter(
            COLOR_FORMAT,
            datefmt=DATE_FORMAT,
            log_colors=LOG_COLORS,
        )
    else:
        formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler (optional)
    file_path = log_file or os.getenv("LOG_FILE", "")
    if file_path:
        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
        fh = RotatingFileHandler(
            file_path, maxBytes=10 * 1024 * 1024, backupCount=5
        )
        fh.setLevel(numeric_level)
        fh.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
        logger.addHandler(fh)

    _LOGGERS[name] = logger
    return logger
