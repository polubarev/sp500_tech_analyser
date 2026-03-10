from __future__ import annotations

import logging
import os


DEFAULT_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def resolve_log_level(value: str | None = None) -> int:
    candidate = (value or os.getenv("SP500_TECH_LOG_LEVEL", "INFO")).upper()
    return getattr(logging, candidate, logging.INFO)


def configure_logging(level: str | None = None) -> None:
    logging.basicConfig(level=resolve_log_level(level), format=DEFAULT_LOG_FORMAT, force=True)
