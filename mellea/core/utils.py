"""Utils for Core Library."""

import json
import logging
import os
import sys

import requests


class RESTHandler(logging.Handler):
    """RESTHandler for logging."""

    def __init__(self, api_url, method="POST", headers=None):
        """Initializes a RESTHandler; uses application/json by default."""
        super().__init__()
        self.api_url = api_url
        self.method = method
        self.headers = headers or {"Content-Type": "application/json"}

    def emit(self, record):
        """Attempts to emit a record to FLOG, or silently fails."""
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
    """Logging formatter for JSON."""

    def format(self, record):  # type: ignore
        """Formats record as a JSON serializable object."""
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
    """A nice custom formatter copied from [https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output](Sergey Pleshakov's post on StackOvervlow)."""

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
        """The format fn."""
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%H:%M:%S")
        return formatter.format(record)


class FancyLogger:
    """A fancy logger."""

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
