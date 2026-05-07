"""OpenTelemetry logging instrumentation for Mellea.

Provides log export using OpenTelemetry Logs API with OTLP exporter support.

Configuration via environment variables:
- MELLEA_LOG_OTLP: Enable OTLP log exporter (default: false)
- OTEL_EXPORTER_OTLP_LOG_ENDPOINT: Log-specific endpoint (optional, overrides general)
- OTEL_EXPORTER_OTLP_ENDPOINT: General endpoint for all signals (fallback)
- OTEL_SERVICE_NAME: Service name for logs (default: mellea)

Example:
    export MELLEA_LOG_OTLP=true
    export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317

Programmatic usage:
    from mellea.telemetry.logging import get_otlp_log_handler
    import logging

    logger = logging.getLogger("my_logger")
    handler = get_otlp_log_handler()
    if handler:
        logger.addHandler(handler)
        logger.info("This log will be exported via OTLP")
"""

import os
import warnings
from typing import Any

# Try to import OpenTelemetry, but make it optional
try:
    from opentelemetry._logs import set_logger_provider
    from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
    from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
    from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
    from opentelemetry.sdk.resources import Resource

    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False


def _setup_logger_provider() -> Any:
    """Set up the LoggerProvider with OTLP exporter.

    Reads endpoint and service-name env vars at call time so that env changes
    made after module import (e.g. in tests) are respected.

    Returns:
        LoggerProvider instance or None if OpenTelemetry is not available or
        no endpoint is configured.
    """
    if not _OTEL_AVAILABLE:
        return None

    endpoint = os.getenv("OTEL_EXPORTER_OTLP_LOG_ENDPOINT") or os.getenv(
        "OTEL_EXPORTER_OTLP_ENDPOINT"
    )
    if not endpoint:
        warnings.warn(
            "OTLP logs exporter is enabled (MELLEA_LOG_OTLP=true) but no endpoint is configured. "
            "Set OTEL_EXPORTER_OTLP_LOG_ENDPOINT or OTEL_EXPORTER_OTLP_ENDPOINT to export logs.",
            UserWarning,
            stacklevel=3,
        )
        return None

    service_name = os.getenv("OTEL_SERVICE_NAME", "mellea")
    resource = Resource.create({"service.name": service_name})  # type: ignore
    logger_provider = LoggerProvider(resource=resource)  # type: ignore

    try:
        otlp_exporter = OTLPLogExporter(endpoint=endpoint)  # type: ignore
        logger_provider.add_log_record_processor(
            BatchLogRecordProcessor(otlp_exporter)  # type: ignore
        )
    except Exception as e:
        warnings.warn(
            f"Failed to initialize OTLP logs exporter: {e}. "
            "Logs will not be exported via OTLP.",
            UserWarning,
            stacklevel=3,
        )
        return None

    set_logger_provider(logger_provider)  # type: ignore
    return logger_provider


# Lazily initialised on first call to get_otlp_log_handler().
_logger_provider: Any = None
_logger_provider_initialised: bool = False


def get_otlp_log_handler() -> Any:
    """Get an OTLP logging handler for Python's logging module.

    The logger provider is initialised on the first call so that environment
    variables set after module import (e.g. ``MELLEA_LOG_OTLP``) are
    respected without requiring a module reload.

    Returns:
        LoggingHandler instance if OTLP logging is enabled and configured,
        None otherwise.

    Example:
        import logging
        from mellea.telemetry.logging import get_otlp_log_handler

        logger = logging.getLogger("my_app")
        handler = get_otlp_log_handler()
        if handler:
            logger.addHandler(handler)
            logger.info("This log will be exported via OTLP")
    """
    global _logger_provider, _logger_provider_initialised
    if not _logger_provider_initialised:
        _logger_provider_initialised = True
        logs_otlp = _OTEL_AVAILABLE and os.getenv(
            "MELLEA_LOG_OTLP", "false"
        ).lower() in ("true", "1", "yes")
        if logs_otlp:
            _logger_provider = _setup_logger_provider()

    if _logger_provider is None:
        return None

    try:
        handler = LoggingHandler(logger_provider=_logger_provider)  # type: ignore
        return handler
    except Exception as e:
        warnings.warn(
            f"Failed to create OTLP logging handler: {e}. "
            "Logs will not be exported via OTLP.",
            UserWarning,
            stacklevel=2,
        )
        return None
