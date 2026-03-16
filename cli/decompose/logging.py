import logging
import sys
from enum import StrEnum


class LogMode(StrEnum):
    demo = "demo"
    debug = "debug"


_CONFIGURED = False


def configure_logging(log_mode: LogMode = LogMode.demo) -> None:
    global _CONFIGURED

    level = logging.DEBUG if log_mode == LogMode.debug else logging.INFO

    root_logger = logging.getLogger()

    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("[%(levelname)s] %(name)s | %(message)s")
        )
        root_logger.handlers.clear()
        root_logger.addHandler(handler)
        _CONFIGURED = True

    root_logger.setLevel(level)

    logging.getLogger("m_decompose").setLevel(level)
    logging.getLogger("mellea").setLevel(level)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_section(logger: logging.Logger, title: str) -> None:
    logger.info("")
    logger.info("=" * 72)
    logger.info(title)
    logger.info("=" * 72)
