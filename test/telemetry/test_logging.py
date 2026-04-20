"""Unit tests for OpenTelemetry logging instrumentation."""

import os
from unittest.mock import MagicMock, patch

import pytest

# Check if OpenTelemetry is available
try:
    from opentelemetry._logs import set_logger_provider
    from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
    from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not OTEL_AVAILABLE, reason="OpenTelemetry not installed"
)


def _reset_logging_modules():
    """Helper to reset logging state and reload modules."""
    import importlib
    import logging

    import mellea.core.utils
    import mellea.telemetry.logging
    from mellea.core.utils import MelleaLogger

    # Clear any existing handlers from previous tests
    fancy_logger = logging.getLogger("fancy_logger")
    fancy_logger.handlers.clear()

    # Reset MelleaLogger singleton
    MelleaLogger.logger = None

    # Force reload of logging module and core.utils to pick up env vars
    importlib.reload(mellea.telemetry.logging)
    importlib.reload(mellea.core.utils)


@pytest.fixture
def clean_logging_env(monkeypatch):
    """Clean logging environment variables before each test."""
    monkeypatch.delenv("MELLEA_LOGS_OTLP", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_SERVICE_NAME", raising=False)

    _reset_logging_modules()
    yield
    _reset_logging_modules()


@pytest.fixture
def enable_otlp_logging(monkeypatch):
    """Enable OTLP logging with endpoint for tests."""
    monkeypatch.setenv("MELLEA_LOGS_OTLP", "true")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")

    _reset_logging_modules()
    yield
    _reset_logging_modules()


# Configuration Tests


def test_otlp_logging_disabled_by_default(clean_logging_env):
    """Test that OTLP logging is disabled by default."""
    from mellea.telemetry.logging import get_otlp_log_handler

    handler = get_otlp_log_handler()
    assert handler is None


def test_otlp_logging_enabled_with_env_var(enable_otlp_logging):
    """Test that OTLP logging can be enabled via environment variable."""
    from mellea.telemetry.logging import get_otlp_log_handler

    handler = get_otlp_log_handler()
    assert handler is not None
    assert isinstance(handler, LoggingHandler)  # type: ignore


def test_otlp_logging_enabled_without_endpoint_warns(monkeypatch, clean_logging_env):
    """Test that enabling OTLP without endpoint produces warning."""
    monkeypatch.setenv("MELLEA_LOGS_OTLP", "true")
    # No endpoint set

    import importlib

    import mellea.telemetry.logging

    with pytest.warns(UserWarning, match="no endpoint is configured"):
        importlib.reload(mellea.telemetry.logging)

    from mellea.telemetry.logging import get_otlp_log_handler

    handler = get_otlp_log_handler()
    assert handler is None


def test_otlp_logging_with_various_truthy_values(monkeypatch, clean_logging_env):
    """Test that various truthy values enable OTLP logging."""
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")

    for value in ["true", "True", "TRUE", "1", "yes", "Yes", "YES"]:
        monkeypatch.setenv("MELLEA_LOGS_OTLP", value)

        import importlib

        import mellea.telemetry.logging

        importlib.reload(mellea.telemetry.logging)

        from mellea.telemetry.logging import get_otlp_log_handler

        handler = get_otlp_log_handler()
        assert handler is not None, f"Failed for value: {value}"


def test_logs_specific_endpoint_takes_precedence(monkeypatch, clean_logging_env):
    """Test that OTEL_EXPORTER_OTLP_LOGS_ENDPOINT takes precedence."""
    monkeypatch.setenv("MELLEA_LOGS_OTLP", "true")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT", "http://localhost:4318/logs")

    import importlib

    import mellea.telemetry.logging

    importlib.reload(mellea.telemetry.logging)

    # Verify the logs-specific endpoint is used
    assert mellea.telemetry.logging._OTLP_LOGS_ENDPOINT == "http://localhost:4318/logs"


# Handler Integration Tests


def test_get_otlp_log_handler_can_be_added_to_logger(enable_otlp_logging):
    """Test that OTLP handler can be added to a Python logger."""
    import logging

    from mellea.telemetry.logging import get_otlp_log_handler

    logger = logging.getLogger("test_logger")
    handler = get_otlp_log_handler()

    assert handler is not None
    logger.addHandler(handler)

    # Verify handler was added
    assert handler in logger.handlers

    # Clean up
    logger.removeHandler(handler)


# MelleaLogger Integration Tests


def test_fancy_logger_includes_otlp_handler_when_enabled(enable_otlp_logging):
    """Test that MelleaLogger includes OTLP handler when enabled."""
    from mellea.core.utils import MelleaLogger

    logger = MelleaLogger.get_logger()

    # Check that logger has handlers
    assert len(logger.handlers) > 0

    # Check if any handler is a LoggingHandler (OTLP)
    has_otlp_handler = any(isinstance(h, LoggingHandler) for h in logger.handlers)  # type: ignore
    assert has_otlp_handler, "MelleaLogger should have OTLP handler when enabled"


def test_fancy_logger_works_without_otlp(clean_logging_env):
    """Test that MelleaLogger works normally when OTLP is disabled."""
    from mellea.core.utils import MelleaLogger

    logger = MelleaLogger.get_logger()

    # Should still have REST and console handlers
    assert len(logger.handlers) >= 2

    # Should not have OTLP handler
    has_otlp_handler = any(isinstance(h, LoggingHandler) for h in logger.handlers)  # type: ignore
    assert not has_otlp_handler, (
        "MelleaLogger should not have OTLP handler when disabled"
    )

    # Verify logger can log messages (backward compatibility)
    logger.info("Test message")
