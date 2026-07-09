"""
Logging utilities for CA-EDT-AHMA.
"""

from __future__ import annotations

print("[PROGRESS] Loaded Backend/app/utils/Anomaly_Health_Monitering/logging_utils.py")
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

import os as _os
import sys as _sys

if __package__ in {None, ""}:
    _backend_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..', '..'))
    if _backend_root not in _sys.path:
        _sys.path.append(_backend_root)

from app.config.Anomaly_Health_Monitering.Config import Config


def get_logger(name: str) -> logging.Logger:
    """
    Create and return a configured logger.

    Args:
        name: Logger name.

    Returns:
        logging.Logger: Configured logger.
    """
    print("[PROGRESS] Entering Backend/app/utils/Anomaly_Health_Monitering/logging_utils.py::get_logger")
    try:
        Config.create_directories()
    except OSError:
        pass

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    log_file: Path = Config.LOG_DIR / "ca_edt_ahma.log"

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)

    try:
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=5_000_000,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
    except OSError:
        pass

    logger.addHandler(stream_handler)

    return logger


if __name__ == "__main__":
    test_logger = get_logger(__name__)
    test_logger.info("Logging utility initialized successfully.")
    print("Logging utility initialized successfully.")
