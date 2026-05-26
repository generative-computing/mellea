import logging
import sys
from enum import StrEnum


class LogMode(StrEnum):
    """Logging verbosity modes used across the CLI and pipeline.

    Attributes:
        demo: Standard readable logs with moderate detail.
        debug: Verbose logs including internal state and intermediate outputs.
    """

    demo = "demo"
    debug = "debug"


_CONFIGURED = False


def configure_logging(log_mode: LogMode = LogMode.demo) -> None:
    """Configures root logging handlers and log levels for the application.

    Initializes a single stdout stream handler on first invocation and updates
    log levels on subsequent calls. This function is safe to call multiple times.

    Args:
        log_mode: Logging verbosity mode controlling the global log level.

    Returns:
        None.
    """
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
    """Returns a logger instance with the given name.

    Args:
        name: Logger name, typically a module or component identifier.

    Returns:
        A configured ``logging.Logger`` instance.
    """
    return logging.getLogger(name)


def log_section(logger: logging.Logger, title: str) -> None:
    """Emits a formatted section header to visually separate log output.

    Args:
        logger: Logger used to emit the section lines.
        title: Section title displayed between separator lines.

    Returns:
        None.
    """
    logger.info("")
    logger.info("=" * 72)
    logger.info(title)
    logger.info("=" * 72)
