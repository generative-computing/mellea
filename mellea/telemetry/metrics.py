"""OpenTelemetry metrics instrumentation for Mellea.

Provides metrics collection using OpenTelemetry Metrics API with support for:
- Counters: Monotonically increasing values (e.g., request counts)
- Histograms: Value distributions (e.g., latency, token counts)
- UpDownCounters: Values that can increase or decrease (e.g., active sessions)

Configuration via environment variables:
- MELLEA_METRICS_ENABLED: Enable/disable metrics collection (default: false)
- MELLEA_METRICS_CONSOLE: Print metrics to console for debugging (default: false)
- OTEL_EXPORTER_OTLP_ENDPOINT: OTLP endpoint for metric export (optional)
- OTEL_SERVICE_NAME: Service name for metrics (default: mellea)

Example usage:
    from mellea.telemetry.metrics import create_counter, create_histogram

    request_counter = create_counter(
        "mellea.requests",
        description="Total number of LLM requests",
        unit="1"
    )
    request_counter.add(1, {"backend": "ollama", "model": "llama2"})

    latency_histogram = create_histogram(
        "mellea.request.duration",
        description="Request latency distribution",
        unit="ms"
    )
    latency_histogram.record(150.5, {"backend": "ollama"})
"""

import os
import warnings
from typing import Any

# Try to import OpenTelemetry, but make it optional
try:
    from opentelemetry import metrics
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
        OTLPMetricExporter,
    )
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import (
        ConsoleMetricExporter,
        PeriodicExportingMetricReader,
    )
    from opentelemetry.sdk.resources import Resource

    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False
    # Provide dummy types for type hints
    metrics = None  # type: ignore

# Configuration from environment variables
_METRICS_ENABLED = _OTEL_AVAILABLE and os.getenv(
    "MELLEA_METRICS_ENABLED", "false"
).lower() in ("true", "1", "yes")
_METRICS_CONSOLE = os.getenv("MELLEA_METRICS_CONSOLE", "false").lower() in (
    "true",
    "1",
    "yes",
)
_OTLP_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
_SERVICE_NAME = os.getenv("OTEL_SERVICE_NAME", "mellea")

# Lazy initialization - MeterProvider created on first metric creation
_meter_provider: Any = None
_meter: Any = None


def _setup_meter_provider() -> Any:
    """Set up the MeterProvider with configured exporters.

    Returns:
        MeterProvider instance or None if OpenTelemetry is not available
    """
    if not _OTEL_AVAILABLE:
        return None

    resource = Resource.create({"service.name": _SERVICE_NAME})  # type: ignore
    readers = []

    # Add OTLP exporter if endpoint is configured
    if _OTLP_ENDPOINT:
        otlp_exporter = OTLPMetricExporter(endpoint=_OTLP_ENDPOINT)  # type: ignore
        readers.append(PeriodicExportingMetricReader(otlp_exporter))  # type: ignore

    # Add console exporter for debugging if enabled
    if _METRICS_CONSOLE:
        try:
            console_exporter = ConsoleMetricExporter()  # type: ignore
            readers.append(PeriodicExportingMetricReader(console_exporter))  # type: ignore
        except Exception:
            # Silently ignore console exporter setup failures
            pass

    # Warn if no exporters are configured
    if not readers:
        warnings.warn(
            "Metrics are enabled (MELLEA_METRICS_ENABLED=true) but no exporters are configured. "
            "Metrics will be collected but not exported. "
            "Set OTEL_EXPORTER_OTLP_ENDPOINT or MELLEA_METRICS_CONSOLE=true to export metrics.",
            UserWarning,
            stacklevel=2,
        )

    provider = MeterProvider(resource=resource, metric_readers=readers)  # type: ignore
    metrics.set_meter_provider(provider)  # type: ignore
    return provider


def _get_meter() -> Any:
    """Get or create the Mellea meter instance.

    Returns:
        Meter instance or None if metrics are disabled
    """
    global _meter_provider, _meter

    if not _METRICS_ENABLED:
        return None

    if _meter is None:
        _meter_provider = _setup_meter_provider()
        if _meter_provider is not None:
            _meter = metrics.get_meter("mellea.metrics", "0.3.0")  # type: ignore

    return _meter


# No-op instrument classes for when metrics are disabled
class _NoOpCounter:
    """No-op counter that does nothing."""

    def add(
        self, amount: int | float, attributes: dict[str, Any] | None = None
    ) -> None:
        """No-op add method."""


class _NoOpHistogram:
    """No-op histogram that does nothing."""

    def record(
        self, amount: int | float, attributes: dict[str, Any] | None = None
    ) -> None:
        """No-op record method."""


class _NoOpUpDownCounter:
    """No-op up-down counter that does nothing."""

    def add(
        self, amount: int | float, attributes: dict[str, Any] | None = None
    ) -> None:
        """No-op add method."""


def create_counter(name: str, description: str = "", unit: str = "1") -> Any:
    """Create a counter instrument for monotonically increasing values.

    Counters are used for values that only increase, such as:
    - Total number of requests
    - Total tokens processed
    - Total errors encountered

    Args:
        name: Metric name (e.g., "mellea.requests.total")
        description: Human-readable description of what this metric measures
        unit: Unit of measurement (e.g., "1" for count, "ms" for milliseconds)

    Returns:
        Counter instrument (or no-op if metrics disabled)

    Example:
        counter = create_counter(
            "mellea.requests.total",
            description="Total LLM requests",
            unit="1"
        )
        counter.add(1, {"backend": "ollama", "status": "success"})
    """
    meter = _get_meter()
    if meter is None:
        return _NoOpCounter()

    return meter.create_counter(name, description=description, unit=unit)


def create_histogram(
    name: str,
    description: str = "",
    unit: str = "1",
) -> Any:
    """Create a histogram instrument for recording value distributions.

    Histograms are used for values that vary and need statistical analysis:
    - Request latency
    - Token counts per request
    - Response sizes

    Args:
        name: Metric name (e.g., "mellea.request.duration")
        description: Human-readable description
        unit: Unit of measurement (e.g., "ms", "tokens", "bytes")

    Returns:
        Histogram instrument (or no-op if metrics disabled)

    Example:
        histogram = create_histogram(
            "mellea.request.duration",
            description="Request latency",
            unit="ms"
        )
        histogram.record(150.5, {"backend": "ollama", "model": "llama2"})
    """
    meter = _get_meter()
    if meter is None:
        return _NoOpHistogram()

    return meter.create_histogram(name, description=description, unit=unit)


def create_up_down_counter(name: str, description: str = "", unit: str = "1") -> Any:
    """Create an up-down counter for values that can increase or decrease.

    UpDownCounters are used for values that go up and down:
    - Active sessions
    - Items in a queue
    - Memory usage

    Args:
        name: Metric name (e.g., "mellea.sessions.active")
        description: Human-readable description
        unit: Unit of measurement

    Returns:
        UpDownCounter instrument (or no-op if metrics disabled)

    Example:
        counter = create_up_down_counter(
            "mellea.sessions.active",
            description="Number of active sessions",
            unit="1"
        )
        counter.add(1)   # Session started
        counter.add(-1)  # Session ended
    """
    meter = _get_meter()
    if meter is None:
        return _NoOpUpDownCounter()

    return meter.create_up_down_counter(name, description=description, unit=unit)


def is_metrics_enabled() -> bool:
    """Check if metrics collection is enabled.

    Returns:
        True if metrics are enabled, False otherwise
    """
    return _METRICS_ENABLED


__all__ = [
    "create_counter",
    "create_histogram",
    "create_up_down_counter",
    "is_metrics_enabled",
]
