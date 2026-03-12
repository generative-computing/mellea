"""Logging utilities for the mellea core library.

Provides ``FancyLogger``, a singleton logger with colour-coded console output and
an optional REST handler (``RESTHandler``) that forwards log records to a local
``/api/receive`` endpoint when the ``FLOG`` environment variable is set. All
internal mellea modules obtain their logger via ``FancyLogger.get_logger()``.
"""

import json
import logging
import os
import sys

import requests


class RESTHandler(logging.Handler):
    """Logging handler that forwards records to a local REST endpoint.

    Sends log records as JSON to ``/api/receive`` when the ``FLOG`` environment
    variable is set. Failures are silently suppressed to avoid disrupting the
    application.

    Attributes:
        api_url (str): The URL of the REST endpoint that receives log records.
        method (str): The HTTP method used when sending records (default ``"POST"``).
        headers (dict): HTTP headers sent with each request; defaults to ``{"Content-Type": "application/json"}``.
    """

    def __init__(self, api_url, method="POST", headers=None):
        """Initializes a RESTHandler; uses application/json by default."""
        super().__init__()
        self.api_url = api_url
        self.method = method
        self.headers = headers or {"Content-Type": "application/json"}

    def emit(self, record):
        """Forwards a log record to the REST endpoint when the ``FLOG`` environment variable is set.

        Silently suppresses any network or HTTP errors to avoid disrupting the application.

        Args:
            record (logging.LogRecord): The log record to forward.
        """
        if os.environ.get("FLOG"):
            log_data = self.format(record)
            try:
                response = requests.request(
                    self.method,
                    self.api_url,
                    headers=self.headers,
                    # data=json.dumps([{"log": log_data}]),
                    data=json.dumps([log_data]),
                )
                response.raise_for_status()
            except requests.exceptions.RequestException as _:
                pass


class JsonFormatter(logging.Formatter):
    """Logging formatter that serialises log records as structured JSON dicts.

    Includes timestamp, level, message, module, function name, line number,
    process ID, thread ID, and (if present) exception information.
    """

    def format(self, record):  # type: ignore
        """Formats a log record as a JSON-serialisable dictionary.

        Includes timestamp, level, message, module, function name, line number,
        process ID, thread ID, and exception info if present.

        Args:
            record (logging.LogRecord): The log record to format.
        """
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line_number": record.lineno,
            "process_id": record.process,
            "thread_id": record.thread,
        }
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        return log_record


class CustomFormatter(logging.Formatter):
    """A nice custom formatter copied from [https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output](Sergey Pleshakov's post on StackOverflow).

    Attributes:
        cyan (str): ANSI escape code for cyan text, used for DEBUG messages.
        grey (str): ANSI escape code for grey text, used for INFO messages.
        yellow (str): ANSI escape code for yellow text, used for WARNING messages.
        red (str): ANSI escape code for red text, used for ERROR messages.
        bold_red (str): ANSI escape code for bold red text, used for CRITICAL messages.
        reset (str): ANSI escape code to reset text colour.
        FORMATS (dict): Mapping from logging level integer to the colour-formatted format string.
    """

    cyan = "\033[96m"  # Cyan
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    _format_string = "=== %(asctime)s-%(levelname)s ======\n%(message)s"

    FORMATS = {
        logging.DEBUG: cyan + _format_string + reset,
        logging.INFO: grey + _format_string + reset,
        logging.WARNING: yellow + _format_string + reset,
        logging.ERROR: red + _format_string + reset,
        logging.CRITICAL: bold_red + _format_string + reset,
    }

    def format(self, record):
        """Formats a log record using a colour-coded ANSI format string based on the record's log level.

        Args:
            record (logging.LogRecord): The log record to format.
        """
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%H:%M:%S")
        return formatter.format(record)


class FancyLogger:
    """Singleton logger with colour-coded console output and optional REST forwarding.

    Obtain the shared logger instance via ``FancyLogger.get_logger()``. Log level
    defaults to ``INFO`` but can be raised to ``DEBUG`` by setting the ``DEBUG``
    environment variable. When the ``FLOG`` environment variable is set, records are
    also forwarded to a local ``/api/receive`` REST endpoint via ``RESTHandler``.

    Attributes:
        logger (logging.Logger | None): The shared ``logging.Logger`` instance; ``None`` until first call to ``get_logger()``.
        CRITICAL (int): Numeric level for critical log messages (50).
        FATAL (int): Alias for ``CRITICAL`` (50).
        ERROR (int): Numeric level for error log messages (40).
        WARNING (int): Numeric level for warning log messages (30).
        WARN (int): Alias for ``WARNING`` (30).
        INFO (int): Numeric level for informational log messages (20).
        DEBUG (int): Numeric level for debug log messages (10).
        NOTSET (int): Numeric level meaning no level is set (0).
    """

    logger = None

    CRITICAL = 50
    FATAL = CRITICAL
    ERROR = 40
    WARNING = 30
    WARN = WARNING
    INFO = 20
    DEBUG = 10
    NOTSET = 0

    @staticmethod
    def get_logger():
        """Returns a FancyLogger.logger and sets level based upon env vars."""
        if FancyLogger.logger is None:
            logger = logging.getLogger("fancy_logger")
            # Only set default level if user hasn't already configured it
            if logger.level == logging.NOTSET:
                if os.environ.get("DEBUG"):
                    logger.setLevel(FancyLogger.DEBUG)
                else:
                    logger.setLevel(FancyLogger.INFO)

            # Define REST API endpoint
            api_url = "http://localhost:8000/api/receive"

            # Create REST handler
            rest_handler = RESTHandler(api_url)

            # Create formatter and set it for the handler
            # formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            rest_handler.setFormatter(JsonFormatter())

            # Add handler to the logger
            logger.addHandler(rest_handler)

            stream_handler = logging.StreamHandler(stream=sys.stdout)
            # stream_handler.setLevel(logging.INFO)
            stream_handler.setFormatter(CustomFormatter(datefmt="%H:%M:%S,%03d"))
            logger.addHandler(stream_handler)
            FancyLogger.logger = logger
        return FancyLogger.logger
