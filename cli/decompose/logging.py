from enum import StrEnum

try:
    from mellea.logger import get_logger as _mellea_get_logger
except ImportError as e:
    raise ImportError(
        "Failed to import Mellea logger. "
        "Please update m_decompose/logging.py to match your local Mellea logger import path."
    ) from e


class LogMode(StrEnum):
    demo = "demo"
    debug = "debug"


def get_logger(name: str, log_mode: LogMode = LogMode.demo):
    logger = _mellea_get_logger(name)

    # Assumes Mellea logger supports setLevel like stdlib logger.
    if log_mode == LogMode.debug:
        logger.setLevel("DEBUG")
    else:
        logger.setLevel("INFO")

    return logger


def log_section(logger, title: str) -> None:
    logger.info("")
    logger.info("=" * 72)
    logger.info(title)
    logger.info("=" * 72)