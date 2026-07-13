"""Shared test helpers for the `mellea.telemetry` submodules."""

import os

import pytest

_TELEMETRY_DIR = os.path.dirname(__file__)


def pytest_collection_modifyitems(items):
    """Ignore the duplicate-registration warning that the reset helpers provoke.

    The hook receives the whole session's items even from a subdir conftest, so
    scope the marker to this directory by path.
    """
    for item in items:
        if str(item.fspath).startswith(_TELEMETRY_DIR):
            item.add_marker(pytest.mark.filterwarnings("ignore:.*already registered"))


def reset_metrics_state() -> None:
    """Reset metrics module state and re-run setup so env-var changes take effect."""
    from mellea.telemetry import metrics

    if metrics._meter_provider is not None:
        metrics._meter_provider.shutdown()
    metrics._meter_provider = None
    metrics._meter = None
    # Cached lazy instruments — bound to the old MeterProvider, must clear so
    # the next record_* call re-creates them against the new provider.
    metrics._input_token_counter = None
    metrics._output_token_counter = None
    metrics._duration_histogram = None
    metrics._ttfb_histogram = None
    metrics._error_counter = None
    metrics._cost_counter = None
    metrics._sampling_attempts_counter = None
    metrics._sampling_successes_counter = None
    metrics._sampling_failures_counter = None
    metrics._requirement_checks_counter = None
    metrics._requirement_failures_counter = None
    metrics._tool_calls_counter = None
    # Re-register: another test's shutdown_plugins() may have emptied the manager.
    metrics._plugins_registered = False
    metrics._setup_metrics()


def reset_tracing_state() -> None:
    """Reset tracing module state and re-run setup so env-var changes take effect."""
    from mellea.telemetry import tracing

    if tracing._tracer_provider is not None:
        tracing._tracer_provider.shutdown()
    tracing._tracer_provider = None
    tracing._application_tracer = None
    tracing._backend_tracer = None
    tracing._in_flight_spans.clear()
    tracing._reattached_tokens.clear()
    # Re-register: another test's shutdown_plugins() may have emptied the manager.
    tracing._plugins_registered = False
    tracing._setup_tracing()


def reset_logging_state() -> None:
    """Reset logging module state and the MelleaLogger singleton.

    The logging module's lazy init re-runs on the next `get_otlp_log_handler()`
    call. Tests that want to capture the OTLP warning or intercept the exporter
    construction must do so inside the block where they invoke
    `get_otlp_log_handler()`.
    """
    import logging

    from mellea.core.utils import MelleaLogger
    from mellea.telemetry import logging as mlogging

    logging.getLogger("mellea").handlers.clear()
    MelleaLogger.logger = None
    mlogging._logger_provider = None
    mlogging._logger_provider_initialised = False


def reset_pricing_state() -> None:
    """Reset pricing module state and re-run setup."""
    from mellea.telemetry import pricing

    pricing._warned_models.clear()
    pricing._setup_pricing()
