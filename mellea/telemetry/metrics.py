"""OpenTelemetry metrics instrumentation for Mellea.

Provides metrics collection using OpenTelemetry Metrics API with support for:
- Counters: Monotonically increasing values (e.g., request counts)
- Histograms: Value distributions (e.g., latency, token counts)
- UpDownCounters: Values that can increase or decrease (e.g., active sessions)

Configuration via environment variables:
- MELLEA_METRICS_ENABLED: Enable/disable metrics collection (default: false)
- MELLEA_METRICS_CONSOLE: Print metrics to console for debugging (default: false)
- MELLEA_METRICS_OTLP: Enable OTLP metrics exporter (default: false)
- OTEL_EXPORTER_OTLP_ENDPOINT: OTLP endpoint for metric export (optional)
- OTEL_EXPORTER_OTLP_METRICS_ENDPOINT: OTLP metrics-specific endpoint (optional, overrides OTEL_EXPORTER_OTLP_ENDPOINT)
- MELLEA_METRICS_EXPORT_INTERVAL: Export interval in seconds (default: 60)
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
from importlib.metadata import version
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
_METRICS_OTLP = os.getenv("MELLEA_METRICS_OTLP", "false").lower() in (
    "true",
    "1",
    "yes",
)
# Metrics-specific endpoint takes precedence over general OTLP endpoint
_OTLP_METRICS_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT") or os.getenv(
    "OTEL_EXPORTER_OTLP_ENDPOINT"
)
_SERVICE_NAME = os.getenv("OTEL_SERVICE_NAME", "mellea")

# Parse export interval (default 60 seconds)
try:
    _EXPORT_INTERVAL_SECONDS = int(os.getenv("MELLEA_METRICS_EXPORT_INTERVAL", "60"))
    if _EXPORT_INTERVAL_SECONDS <= 0:
        warnings.warn(
            f"Invalid MELLEA_METRICS_EXPORT_INTERVAL value: {_EXPORT_INTERVAL_SECONDS}. "
            "Must be positive. Using default of 60 seconds.",
            UserWarning,
            stacklevel=2,
        )
        _EXPORT_INTERVAL_SECONDS = 60
except ValueError:
    warnings.warn(
        f"Invalid MELLEA_METRICS_EXPORT_INTERVAL value: {os.getenv('MELLEA_METRICS_EXPORT_INTERVAL')}. "
        "Must be an integer. Using default of 60 seconds.",
        UserWarning,
        stacklevel=2,
    )
    _EXPORT_INTERVAL_SECONDS = 60

# Convert to milliseconds for OpenTelemetry
_EXPORT_INTERVAL_MILLIS = _EXPORT_INTERVAL_SECONDS * 1000


def _setup_meter_provider() -> Any:
    """Set up the MeterProvider with configured exporters.

    Returns:
        MeterProvider instance or None if OpenTelemetry is not available
    """
    if not _OTEL_AVAILABLE:
        return None

    resource = Resource.create({"service.name": _SERVICE_NAME})  # type: ignore
    readers = []

    # Add OTLP exporter if explicitly enabled
    if _METRICS_OTLP:
        if _OTLP_METRICS_ENDPOINT:
            try:
                otlp_exporter = OTLPMetricExporter(  # type: ignore
                    endpoint=_OTLP_METRICS_ENDPOINT
                )
                readers.append(
                    PeriodicExportingMetricReader(  # type: ignore
                        otlp_exporter, export_interval_millis=_EXPORT_INTERVAL_MILLIS
                    )
                )
            except Exception as e:
                warnings.warn(
                    f"Failed to initialize OTLP metrics exporter: {e}. "
                    "Metrics will not be exported via OTLP.",
                    UserWarning,
                    stacklevel=2,
                )
        else:
            warnings.warn(
                "OTLP metrics exporter is enabled (MELLEA_METRICS_OTLP=true) but no endpoint is configured. "
                "Set OTEL_EXPORTER_OTLP_METRICS_ENDPOINT or OTEL_EXPORTER_OTLP_ENDPOINT to export metrics.",
                UserWarning,
                stacklevel=2,
            )

    # Add console exporter for debugging if enabled
    if _METRICS_CONSOLE:
        try:
            console_exporter = ConsoleMetricExporter()  # type: ignore
            readers.append(
                PeriodicExportingMetricReader(  # type: ignore
                    console_exporter, export_interval_millis=_EXPORT_INTERVAL_MILLIS
                )
            )
        except Exception as e:
            warnings.warn(
                f"Failed to initialize console metrics exporter: {e}. "
                "Metrics will not be printed to console.",
                UserWarning,
                stacklevel=2,
            )

    # Warn if no exporters are configured
    if not readers:
        warnings.warn(
            "Metrics are enabled (MELLEA_METRICS_ENABLED=true) but no exporters are configured. "
            "Metrics will be collected but not exported. "
            "Set MELLEA_METRICS_OTLP=true with an endpoint (OTEL_EXPORTER_OTLP_METRICS_ENDPOINT or "
            "OTEL_EXPORTER_OTLP_ENDPOINT) or set MELLEA_METRICS_CONSOLE=true to export metrics.",
            UserWarning,
            stacklevel=2,
        )

    provider = MeterProvider(resource=resource, metric_readers=readers)  # type: ignore
    metrics.set_meter_provider(provider)  # type: ignore
    return provider


# Initialize meter provider if metrics are enabled
_meter_provider = None
_meter = None

if _OTEL_AVAILABLE and _METRICS_ENABLED:
    _meter_provider = _setup_meter_provider()
    if _meter_provider is not None:
        _meter = metrics.get_meter("mellea.metrics", version("mellea"))  # type: ignore


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
    if _meter is None:
        return _NoOpCounter()

    return _meter.create_counter(name, description=description, unit=unit)


def create_histogram(name: str, description: str = "", unit: str = "1") -> Any:
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
    if _meter is None:
        return _NoOpHistogram()

    return _meter.create_histogram(name, description=description, unit=unit)


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
    if _meter is None:
        return _NoOpUpDownCounter()

    return _meter.create_up_down_counter(name, description=description, unit=unit)


def is_metrics_enabled() -> bool:
    """Check if metrics collection is enabled.

    Returns:
        True if metrics are enabled, False otherwise
    """
    return _METRICS_ENABLED


# Token usage counters following Gen-AI semantic conventions
# These are lazily initialized on first use and kept internal
_input_token_counter: Any = None
_output_token_counter: Any = None


def _get_token_counters() -> tuple[Any, Any]:
    """Get or create token usage counters (internal use only).

    Returns:
        Tuple of (input_counter, output_counter)
    """
    global _input_token_counter, _output_token_counter

    if _input_token_counter is None:
        _input_token_counter = create_counter(
            "mellea.llm.tokens.input",
            description="Total number of input tokens processed by LLM",
            unit="tokens",
        )

    if _output_token_counter is None:
        _output_token_counter = create_counter(
            "mellea.llm.tokens.output",
            description="Total number of output tokens generated by LLM",
            unit="tokens",
        )

    return _input_token_counter, _output_token_counter


def record_token_usage_metrics(
    input_tokens: int | None,
    output_tokens: int | None,
    model: str,
    backend: str,
    system: str,
) -> None:
    """Record token usage metrics following Gen-AI semantic conventions.

    This is a no-op when metrics are disabled, ensuring zero overhead.

    Args:
        input_tokens: Number of input tokens (prompt tokens), or None if unavailable
        output_tokens: Number of output tokens (completion tokens), or None if unavailable
        model: Model identifier (e.g., "gpt-4", "llama2:7b")
        backend: Backend class name (e.g., "OpenAIBackend", "OllamaBackend")
        system: Gen-AI system name (e.g., "openai", "ollama", "watsonx")

    Example:
        record_token_usage_metrics(
            input_tokens=150,
            output_tokens=50,
            model="llama2:7b",
            backend="OllamaBackend",
            system="ollama"
        )
    """
    # Early return if metrics are disabled (zero overhead)
    if not _METRICS_ENABLED:
        return

    # Get the token counters (lazily initialized)
    input_counter, output_counter = _get_token_counters()

    # Prepare attributes following Gen-AI semantic conventions
    attributes = {
        "gen_ai.system": system,
        "gen_ai.request.model": model,
        "mellea.backend": backend,
    }

    # Record input tokens if available
    if input_tokens is not None and input_tokens > 0:
        input_counter.add(input_tokens, attributes)

    # Record output tokens if available
    if output_tokens is not None and output_tokens > 0:
        output_counter.add(output_tokens, attributes)


__all__ = [
    "create_counter",
    "create_histogram",
    "create_up_down_counter",
    "is_metrics_enabled",
    "record_token_usage_metrics",
]
