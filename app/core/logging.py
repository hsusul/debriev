"""Logging helpers."""

import logging


def configure_logging(level: str = "INFO") -> None:
    """Configure a minimal process-wide logging setup."""

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

