"""Logging utilities for the mellea core library.

Provides ``MelleaLogger``, a singleton logger with colour-coded console output and
an optional REST handler (``RESTHandler``) that forwards log records to a local
``/api/receive`` endpoint when the ``MELLEA_FLOG`` environment variable is set. All
internal mellea modules obtain their logger via ``MelleaLogger.get_logger()``.

Environment variables
---------------------
``MELLEA_LOG_LEVEL``
    Minimum log level name (e.g. ``DEBUG``, ``INFO``, ``WARNING``).  Defaults to
    ``INFO``.
``MELLEA_LOG_JSON``
    Set to any truthy value (``1``, ``true``, ``yes``) to emit structured JSON on
    the console instead of the colour-coded human-readable format.
``MELLEA_FLOG``
    When set, log records are forwarded to a local REST endpoint.
"""

import contextlib
import contextvars
import json
import logging
import os
import sys
import threading
from collections.abc import Generator
from typing import Any

import requests

try:
    from opentelemetry import trace as otel_trace

    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False

# ---------------------------------------------------------------------------
# Per-task/coroutine context fields (safe for asyncio — each Task gets its own copy)
# ---------------------------------------------------------------------------
_log_context: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "log_context_fields", default={}
)

# Lock used to make MelleaLogger singleton initialisation thread-safe.
_logger_lock: threading.Lock = threading.Lock()

# Standard LogRecord attribute names that must not be overwritten by callers.
RESERVED_LOG_RECORD_ATTRS: frozenset[str] = frozenset(
    (
        "args",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "message",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
    )
)


def set_log_context(**fields: Any) -> None:
    """Inject extra fields into every log record emitted from this coroutine or thread.

    Call this at the start of a request or task to attach identifiers such as
    ``trace_id`` or ``request_id`` without modifying individual log calls.

    .. note::
        Prefer :func:`log_context` as the primary API — it guarantees cleanup
        (including restoring outer values on same-key nesting) even on
        exceptions.

    Args:
        **fields: Arbitrary key-value pairs to include in log records.

    Raises:
        ValueError: If any key clashes with a standard ``logging.LogRecord``
            attribute (e.g. ``levelname``, ``module``, ``thread``).
    """
    invalid = frozenset(fields) & RESERVED_LOG_RECORD_ATTRS
    if invalid:
        raise ValueError(
            f"Context field names clash with LogRecord reserved attributes: "
            f"{sorted(invalid)}.  Choose different names."
        )
    _log_context.set({**_log_context.get(), **fields})


def clear_log_context() -> None:
    """Remove all context fields set by :func:`set_log_context` for this coroutine/thread."""
    _log_context.set({})


@contextlib.contextmanager
def log_context(**fields: Any) -> Generator[None, None, None]:
    """Context manager that injects *fields* for the duration of the block.

    On exit — including on exceptions — the context is restored to its state
    before the block via a ``ContextVar`` token.  This is safe for both nested
    usage and concurrent asyncio tasks: each ``asyncio.Task`` owns an isolated
    copy of the context variable, so coroutines running on the same event-loop
    thread cannot overwrite each other's fields.

    Example::

        with log_context(trace_id="abc-123", request_id="req-1"):
            logger.info("Handling request")   # both IDs appear here
        logger.info("After request")          # IDs are gone

    Args:
        **fields: Key-value pairs to inject.  Same restrictions as
            :func:`set_log_context` — reserved ``LogRecord`` attribute names
            are rejected with ``ValueError``.

    Yields:
        None.  The manager is used only for its enter/exit side effects.

    Raises:
        ValueError: If any key clashes with a reserved ``LogRecord`` attribute.
    """
    invalid = frozenset(fields) & RESERVED_LOG_RECORD_ATTRS
    if invalid:
        raise ValueError(
            f"Context field names clash with LogRecord reserved attributes: "
            f"{sorted(invalid)}.  Choose different names."
        )
    token = _log_context.set({**_log_context.get(), **fields})
    try:
        yield
    finally:
        _log_context.reset(token)


class ContextFilter(logging.Filter):
    """Logging filter that injects async-safe ContextVar fields into every record.

    Fields registered via :func:`set_log_context` are copied onto the
    ``logging.LogRecord`` before formatters see it, enabling trace/request IDs
    to appear in structured output without touching call sites.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Attach async-safe ContextVar fields to *record* and allow it through.

        Args:
            record (logging.LogRecord): The log record being processed.

        Returns:
            bool: Always ``True`` — the record is never suppressed.
        """
        fields: dict[str, Any] = _log_context.get()
        for key, value in fields.items():
            setattr(record, key, value)
        return True


class OtelTraceFilter(logging.Filter):
    """Logging filter that injects the current OpenTelemetry trace context into log records.

    Adds ``trace_id`` and ``span_id`` attributes (hex strings) to every
    ``LogRecord`` when an active span exists.  When OpenTelemetry is not
    installed the filter is a true no-op: it adds no attributes and takes no
    branches, so there is zero overhead on the hot logging path.  Formatters
    use ``hasattr`` / ``getattr`` to handle the absent attributes gracefully.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Adds trace_id and span_id to the log record from the current OTel span.

        No-op when OpenTelemetry is not installed or when there is no active span.

        Args:
            record (logging.LogRecord): The log record to enrich.

        Returns:
            bool: Always ``True`` — the record is never suppressed.
        """
        if _OTEL_AVAILABLE:
            ctx = otel_trace.get_current_span().get_span_context()
            if ctx.is_valid:
                record.trace_id = format(ctx.trace_id, "032x")
                record.span_id = format(ctx.span_id, "016x")
        return True


class RESTHandler(logging.Handler):
    """Logging handler that forwards records to a local REST endpoint.

    Sends log records as JSON to ``/api/receive`` when the ``MELLEA_FLOG`` environment
    variable is set. Failures are silently suppressed to avoid disrupting the
    application.

    Args:
        api_url (str): The URL of the REST endpoint that receives log records.
        method (str): HTTP method to use when sending records (default ``"POST"``).
        headers (dict | None): HTTP headers to send; defaults to
            ``{"Content-Type": "application/json"}`` when ``None``.
    """

    def __init__(
        self, api_url: str, method: str = "POST", headers: dict[str, str] | None = None
    ) -> None:
        """Initializes a RESTHandler; uses application/json by default."""
        super().__init__()
        self.api_url = api_url
        self.method = method
        self.headers = headers or {"Content-Type": "application/json"}

    def emit(self, record: logging.LogRecord) -> None:
        """Forwards a log record to the REST endpoint when the ``MELLEA_FLOG`` environment variable is set.

        Silently suppresses any network or HTTP errors to avoid disrupting the application.

        Args:
            record (logging.LogRecord): The log record to forward.
        """
        if _check_flog_env():
            formatter = self.formatter
            if isinstance(formatter, JsonFormatter):
                log_dict = formatter.format_as_dict(record)
            else:
                log_dict = {"message": self.format(record)}
            try:
                response = requests.request(
                    self.method,
                    self.api_url,
                    headers=self.headers,
                    data=json.dumps([log_dict]),
                )
                response.raise_for_status()
            except requests.exceptions.RequestException as _:
                pass


class JsonFormatter(logging.Formatter):
    """Logging formatter that serialises log records as structured JSON strings.

    Produces a consistent JSON schema with a fixed set of core fields.
    Additional fields can be injected at construction time (``extra_fields``) or
    dynamically per-thread via :func:`set_log_context` / :class:`ContextFilter`.
    Includes trace_id and span_id when OpenTelemetry tracing is active.

    Args:
        timestamp_format: ``strftime`` format for the ``timestamp`` field.
            Defaults to ISO-8601 (``"%Y-%m-%dT%H:%M:%S"``).
        include_fields: Whitelist of **core** field names to keep.  When ``None``
            all core fields are included.  Note: this filter applies only to the
            fields listed in ``_DEFAULT_FIELDS``; ``extra_fields`` passed to the
            constructor and dynamic context fields (set via
            :func:`set_log_context`) are **always** included regardless of this
            setting.
        exclude_fields: Set of core field names to drop.  Applied after
            *include_fields*.
        extra_fields: Static key-value pairs merged into every log record.

    Attributes:
        _DEFAULT_FIELDS (tuple[str, ...]): Canonical ordered list of core field
            names produced by this formatter.
    """

    _DEFAULT_FIELDS: tuple[str, ...] = (
        "timestamp",
        "level",
        "message",
        "module",
        "function",
        "line_number",
        "process_id",
        "thread_id",
    )

    def __init__(
        self,
        timestamp_format: str = "%Y-%m-%dT%H:%M:%S",
        include_fields: list[str] | None = None,
        exclude_fields: list[str] | None = None,
        extra_fields: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialises the formatter; passes remaining kwargs to ``logging.Formatter``."""
        super().__init__(datefmt=timestamp_format, **kwargs)

        if include_fields is not None:
            unknown = set(include_fields) - set(self._DEFAULT_FIELDS)
            if unknown:
                raise ValueError(
                    f"include_fields contains unknown field names: {sorted(unknown)}.  "
                    f"Valid fields: {list(self._DEFAULT_FIELDS)}"
                )

        self._include: frozenset[str] | None = (
            frozenset(include_fields) if include_fields is not None else None
        )
        self._exclude: frozenset[str] = frozenset(exclude_fields or [])
        self._extra: dict[str, Any] = dict(extra_fields or {})

    def format_as_dict(self, record: logging.LogRecord) -> dict[str, Any]:
        """Return the log record as a dictionary (public API for external callers).

        Equivalent to :meth:`_build_log_dict` but part of the public interface so
        handlers and other callers do not need to reach into private methods.
        Includes trace_id and span_id when OpenTelemetry tracing is active.

        Args:
            record: The log record to convert.

        Returns:
            A dictionary ready for JSON serialisation.
        """
        return self._build_log_dict(record)

    def _build_log_dict(self, record: logging.LogRecord) -> dict[str, Any]:
        """Build a log record dictionary with core, extra, and context fields.

        Args:
            record: The log record to convert.

        Returns:
            A dictionary ready for JSON serialisation.
        """
        # Build the full set of core fields first.
        # A TypeError here means the caller used %-style format placeholders
        # with the wrong number of arguments (e.g. logger.info("%s %s", one)).
        # Catch it and substitute a safe error string so the record is still emitted.
        try:
            message = record.getMessage()
        except TypeError as exc:
            message = f"<logging format error: {exc}> original msg={record.msg!r}"

        all_core: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": message,
            "module": record.module,
            "function": record.funcName,
            "line_number": record.lineno,
            "process_id": record.process,
            "thread_id": record.thread,
        }
        # Apply include/exclude filtering
        if self._include is not None:
            log_record: dict[str, Any] = {
                k: v for k, v in all_core.items() if k in self._include
            }
        else:
            log_record = {k: v for k, v in all_core.items() if k not in self._exclude}

        # Add trace context if available
        if hasattr(record, "trace_id"):
            log_record["trace_id"] = record.trace_id  # type: ignore[attr-defined]
            log_record["span_id"] = record.span_id  # type: ignore[attr-defined]

        # Exception info
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)

        # Static extra fields (constructor-level)
        log_record.update(self._extra)

        # Dynamic context fields — prefer record attributes (set by
        # ContextFilter) but fall back to ContextVar storage so the
        # formatter works standalone without a filter attached.
        context_fields: dict[str, Any] = _log_context.get()
        for key, value in context_fields.items():
            log_record[key] = getattr(record, key, value)

        return log_record

    def format(self, record: logging.LogRecord) -> str:
        """Formats a log record as a JSON string.

        Core fields are filtered by *include_fields* / *exclude_fields*.
        Static *extra_fields* and any per-task ContextVar fields (set via
        :func:`set_log_context`) are merged in after the core fields.

        Args:
            record (logging.LogRecord): The log record to format.

        Returns:
            str: A JSON-serialised log record.
        """
        return json.dumps(self._build_log_dict(record), default=str)


class CustomFormatter(logging.Formatter):
    """A nice custom formatter copied from [Sergey Pleshakov's post on StackOverflow](https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output).

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

    def format(self, record: logging.LogRecord) -> str:
        """Formats a log record using a colour-coded ANSI format string based on the record's log level.

        Appends ``[trace_id=… span_id=…]`` when ``OtelTraceFilter`` has
        populated those fields on the record and a trace is active.

        Args:
            record (logging.LogRecord): The log record to format.

        Returns:
            str: The formatted log record string with ANSI colour codes applied.
        """
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%H:%M:%S")
        result = formatter.format(record)
        trace_id = getattr(record, "trace_id", None)
        if trace_id is not None:
            result += f" [trace_id={trace_id} span_id={record.span_id}]"  # type: ignore[attr-defined]
        return result


class MelleaLogger:
    """Singleton logger with colour-coded console output and optional REST forwarding.

    Obtain the shared logger instance via ``MelleaLogger.get_logger()``. Log level
    defaults to ``INFO`` but can be overridden via ``MELLEA_LOG_LEVEL``. When the
    ``MELLEA_FLOG`` environment variable is set, records are also forwarded to a
    local ``/api/receive`` REST endpoint via ``RESTHandler``.

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
    def _resolve_log_level() -> int:
        """Resolves the effective log level from environment variables.

        Checks ``MELLEA_LOG_LEVEL`` and defaults to ``INFO``.

        Returns:
            int: A :mod:`logging` level integer.
        """
        level_name = os.environ.get("MELLEA_LOG_LEVEL", "").strip().upper()
        if level_name:
            numeric = getattr(logging, level_name, None)
            if isinstance(numeric, int):
                return numeric
        return MelleaLogger.INFO

    @staticmethod
    def get_logger() -> logging.Logger:
        """Returns a MelleaLogger.logger and sets level based upon env vars.

        The logger is created once (singleton).  Subsequent calls return the
        cached instance.  Initialisation is protected by a module-level lock so
        concurrent callers at startup cannot create duplicate handlers.

        Returns:
            Configured logger with REST, stream, and optional OTLP handlers.
        """
        if MelleaLogger.logger is None:
            with _logger_lock:
                # Second check inside the lock: another thread may have finished
                # initialisation while we were waiting.
                if MelleaLogger.logger is None:
                    logger = logging.getLogger("fancy_logger")

                    # Attach both filters so they reach all handlers
                    logger.addFilter(ContextFilter())
                    logger.addFilter(OtelTraceFilter())

                    # Only set default level if user hasn't already configured it
                    if logger.level == logging.NOTSET:
                        logger.setLevel(MelleaLogger._resolve_log_level())

                    # --- REST handler ---
                    api_url = "http://localhost:8000/api/receive"
                    rest_handler = RESTHandler(api_url)
                    rest_handler.setFormatter(JsonFormatter())
                    logger.addHandler(rest_handler)

                    # --- Console / stream handler ---
                    stream_handler = logging.StreamHandler(stream=sys.stdout)
                    use_json_console = os.environ.get(
                        "MELLEA_LOG_JSON", ""
                    ).strip().lower() in ("1", "true", "yes")
                    if use_json_console:
                        stream_handler.setFormatter(JsonFormatter())
                    else:
                        stream_handler.setFormatter(
                            CustomFormatter(datefmt="%H:%M:%S,%03d")
                        )
                    logger.addHandler(stream_handler)

                    # --- Optional OTLP handler ---
                    from ..telemetry import get_otlp_log_handler

                    otlp_handler = get_otlp_log_handler()
                    if otlp_handler:
                        otlp_handler.setFormatter(JsonFormatter())
                        logger.addHandler(otlp_handler)

                    MelleaLogger.logger = logger
        return MelleaLogger.logger


def _check_flog_env() -> bool:
    """Check MELLEA_FLOG, with a DeprecationWarning fallback for the old FLOG name."""
    if os.environ.get("MELLEA_FLOG"):
        return True
    if os.environ.get("FLOG"):
        import warnings

        warnings.warn(
            "The FLOG environment variable is deprecated and will be removed in a future release. "
            "Use MELLEA_FLOG instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return True
    return False
