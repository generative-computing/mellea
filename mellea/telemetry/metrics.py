"""OpenTelemetry metrics instrumentation for Mellea.

Provides metrics collection using OpenTelemetry Metrics API with support for:
- Counters: Monotonically increasing values (e.g., request counts, token usage)
- Histograms: Value distributions (e.g., latency, token counts)
- UpDownCounters: Values that can increase or decrease (e.g., active sessions)

Metrics Exporters:
- Console: Print metrics to console for debugging
- OTLP: Export to OpenTelemetry Protocol collectors (Jaeger, Grafana, etc.)
- Prometheus: Register metrics with prometheus_client registry for scraping

Configuration via environment variables:

General:
- MELLEA_METRICS_ENABLED: Enable/disable metrics collection (default: false)
- OTEL_SERVICE_NAME: Service name for metrics (default: mellea)

Console Exporter (debugging):
- MELLEA_METRICS_CONSOLE: Print metrics to console (default: false)

OTLP Exporter (production observability):
- MELLEA_METRICS_OTLP: Enable OTLP metrics exporter (default: false)
- OTEL_EXPORTER_OTLP_ENDPOINT: OTLP endpoint for all signals (optional)
- OTEL_EXPORTER_OTLP_METRICS_ENDPOINT: Metrics-specific endpoint (optional, overrides general)
- OTEL_METRIC_EXPORT_INTERVAL: Export interval in milliseconds (default: 60000)

Prometheus Exporter:
- MELLEA_METRICS_PROMETHEUS: Enable Prometheus metric reader (default: false)

Pricing (for cost counter):
- MELLEA_PRICING_FILE: Path to a JSON file with custom model pricing overrides (optional)

Multiple exporters can be enabled simultaneously.

Example - Console debugging:
    export MELLEA_METRICS_ENABLED=true
    export MELLEA_METRICS_CONSOLE=true

Example - OTLP production:
    export MELLEA_METRICS_ENABLED=true
    export MELLEA_METRICS_OTLP=true
    export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317

Example - Prometheus monitoring:
    export MELLEA_METRICS_ENABLED=true
    export MELLEA_METRICS_PROMETHEUS=true

Example - Multiple exporters:
    export MELLEA_METRICS_ENABLED=true
    export MELLEA_METRICS_CONSOLE=true
    export MELLEA_METRICS_OTLP=true
    export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
    export MELLEA_METRICS_PROMETHEUS=true

Built-in metrics (auto-recorded via plugins when metrics are enabled):
- Token counters: mellea.llm.tokens.input, mellea.llm.tokens.output (unit: tokens)
- Latency histograms: mellea.llm.request.duration (unit: s), mellea.llm.ttfb (unit: s, streaming only)
- Error counter: mellea.llm.errors (unit: {error}), categorized by semantic error type
- Cost counter: mellea.llm.cost.usd (unit: USD), estimated cost when pricing data is available
- Sampling counters: mellea.sampling.attempts, mellea.sampling.successes, mellea.sampling.failures (unit: {attempt}/{sample}/{failure})
- Requirement counters: mellea.requirement.checks (unit: {check}), mellea.requirement.failures (unit: {failure})
- Tool counter: mellea.tool.calls (unit: {call}), tagged by tool name and status

Programmatic usage:
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
        unit="s"
    )
    latency_histogram.record(1.5, {"backend": "ollama"})
"""

import asyncio
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
    from opentelemetry.sdk.metrics.view import ExplicitBucketHistogramAggregation, View
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
_METRICS_PROMETHEUS = os.getenv("MELLEA_METRICS_PROMETHEUS", "false").lower() in (
    "true",
    "1",
    "yes",
)
_SERVICE_NAME = os.getenv("OTEL_SERVICE_NAME", "mellea")

# Parse export interval (default 60000 milliseconds = 60 seconds)
try:
    _EXPORT_INTERVAL_MILLIS = int(os.getenv("OTEL_METRIC_EXPORT_INTERVAL", "60000"))
    if _EXPORT_INTERVAL_MILLIS <= 0:
        warnings.warn(
            f"Invalid OTEL_METRIC_EXPORT_INTERVAL value: {_EXPORT_INTERVAL_MILLIS}. "
            "Must be positive. Using default of 60000 milliseconds.",
            UserWarning,
            stacklevel=2,
        )
        _EXPORT_INTERVAL_MILLIS = 60000
except ValueError:
    warnings.warn(
        f"Invalid OTEL_METRIC_EXPORT_INTERVAL value: {os.getenv('OTEL_METRIC_EXPORT_INTERVAL')}. "
        "Must be an integer. Using default of 60000 milliseconds.",
        UserWarning,
        stacklevel=2,
    )
    _EXPORT_INTERVAL_MILLIS = 60000


def _setup_meter_provider() -> Any:
    """Set up the MeterProvider with configured exporters.

    Returns:
        MeterProvider instance or None if OpenTelemetry is not available
    """
    if not _OTEL_AVAILABLE:
        return None

    resource = Resource.create({"service.name": _SERVICE_NAME})  # type: ignore
    readers = []

    # Add Prometheus metric reader if enabled.
    # This registers metrics with the prometheus_client default registry.
    # The application is responsible for exposing the registry (e.g. via
    # prometheus_client.start_http_server() or a framework integration).
    if _METRICS_PROMETHEUS:
        try:
            from opentelemetry.exporter.prometheus import PrometheusMetricReader

            prometheus_reader = PrometheusMetricReader()
            readers.append(prometheus_reader)
        except ImportError:
            warnings.warn(
                "Prometheus exporter is enabled (MELLEA_METRICS_PROMETHEUS=true) "
                "but opentelemetry-exporter-prometheus is not installed. "
                "Install it with: pip install mellea[telemetry]",
                UserWarning,
                stacklevel=2,
            )
        except Exception as e:
            warnings.warn(
                f"Failed to initialize Prometheus metric reader: {e}. "
                "Metrics will not be available via Prometheus.",
                UserWarning,
                stacklevel=2,
            )

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
            "Set MELLEA_METRICS_PROMETHEUS=true, "
            "set MELLEA_METRICS_OTLP=true with an endpoint (OTEL_EXPORTER_OTLP_METRICS_ENDPOINT or "
            "OTEL_EXPORTER_OTLP_ENDPOINT), or set MELLEA_METRICS_CONSOLE=true to export metrics.",
            UserWarning,
            stacklevel=2,
        )

    # Configure explicit bucket boundaries for LLM latency histograms
    views = [
        View(  # type: ignore
            instrument_name="mellea.llm.request.duration",
            aggregation=ExplicitBucketHistogramAggregation(  # type: ignore
                [0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120]
            ),
        ),
        View(  # type: ignore
            instrument_name="mellea.llm.ttfb",
            aggregation=ExplicitBucketHistogramAggregation(  # type: ignore
                [0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10]
            ),
        ),
    ]

    provider = MeterProvider(resource=resource, metric_readers=readers, views=views)  # type: ignore
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
    input_tokens: int | None, output_tokens: int | None, model: str, provider: str
) -> None:
    """Record token usage metrics following OpenTelemetry Gen-AI semantic conventions.

    This is a no-op when metrics are disabled, ensuring zero overhead.

    Args:
        input_tokens: Number of input tokens (prompt tokens), or None if unavailable
        output_tokens: Number of output tokens (completion tokens), or None if unavailable
        model: Model identifier (e.g., "gpt-4", "llama2:7b")
        provider: Provider name (e.g., "openai", "ollama", "watsonx")

    Example:
        record_token_usage_metrics(
            input_tokens=150,
            output_tokens=50,
            model="llama2:7b",
            provider="ollama"
        )
    """
    # Early return if metrics are disabled (zero overhead)
    if not _METRICS_ENABLED:
        return

    # Get the token counters (lazily initialized)
    input_counter, output_counter = _get_token_counters()

    # Prepare attributes following OTel Gen-AI semantic conventions
    attributes = {"gen_ai.provider.name": provider, "gen_ai.request.model": model}

    # Record input tokens if available
    if input_tokens is not None and input_tokens > 0:
        input_counter.add(input_tokens, attributes)

    # Record output tokens if available
    if output_tokens is not None and output_tokens > 0:
        output_counter.add(output_tokens, attributes)


# Latency histograms following Gen-AI semantic conventions
# These are lazily initialized on first use and kept internal
_duration_histogram: Any = None
_ttfb_histogram: Any = None


def _get_latency_histograms() -> tuple[Any, Any]:
    """Get or create latency histograms (internal use only).

    Returns:
        Tuple of (duration_histogram, ttfb_histogram)
    """
    global _duration_histogram, _ttfb_histogram

    if _duration_histogram is None:
        _duration_histogram = create_histogram(
            "mellea.llm.request.duration",
            description="Total LLM request duration",
            unit="s",
        )

    if _ttfb_histogram is None:
        _ttfb_histogram = create_histogram(
            "mellea.llm.ttfb",
            description="Time to first token for streaming LLM requests",
            unit="s",
        )

    return _duration_histogram, _ttfb_histogram


def record_request_duration(
    duration_s: float, model: str, provider: str, streaming: bool = False
) -> None:
    """Record total LLM request duration.

    This is a no-op when metrics are disabled, ensuring zero overhead.

    Args:
        duration_s: Request duration in seconds
        model: Model identifier (e.g., "gpt-4", "llama2:7b")
        provider: Provider name (e.g., "openai", "ollama", "watsonx")
        streaming: Whether the request used streaming mode

    Example:
        record_request_duration(
            duration_s=1.25,
            model="llama2:7b",
            provider="ollama",
            streaming=True,
        )
    """
    if not _METRICS_ENABLED:
        return

    duration_hist, _ = _get_latency_histograms()
    attributes = {
        "gen_ai.request.model": model,
        "gen_ai.provider.name": provider,
        "streaming": streaming,
    }
    duration_hist.record(duration_s, attributes)


def record_ttfb(ttfb_s: float, model: str, provider: str) -> None:
    """Record time-to-first-token for streaming LLM requests.

    This is a no-op when metrics are disabled, ensuring zero overhead.
    Should only be called for streaming requests.

    Args:
        ttfb_s: Time to first token in seconds
        model: Model identifier (e.g., "gpt-4", "llama2:7b")
        provider: Provider name (e.g., "openai", "ollama", "watsonx")

    Example:
        record_ttfb(
            ttfb_s=0.18,
            model="llama2:7b",
            provider="ollama",
        )
    """
    if not _METRICS_ENABLED:
        return

    _, ttfb_hist = _get_latency_histograms()
    attributes = {"gen_ai.request.model": model, "gen_ai.provider.name": provider}
    ttfb_hist.record(ttfb_s, attributes)


# Auto-register metrics plugins when metrics are enabled
if _OTEL_AVAILABLE and _METRICS_ENABLED:
    try:
        from mellea.plugins.registry import register
        from mellea.telemetry.metrics_plugins import _METRICS_PLUGIN_CLASSES

        for _plugin_cls in _METRICS_PLUGIN_CLASSES:
            try:
                register(_plugin_cls())
            except ValueError as e:
                # Already registered (expected during module reloads in tests)
                warnings.warn(
                    f"{_plugin_cls.__name__} already registered: {e}",
                    UserWarning,
                    stacklevel=2,
                )
    except ImportError:
        warnings.warn(
            "Metrics are enabled but the plugin framework is not installed. "
            "Token usage and latency metrics will not be recorded automatically. "
            "Install with: pip install mellea[telemetry]",
            UserWarning,
            stacklevel=2,
        )


# ---------------------------------------------------------------------------
# Error counters
# ---------------------------------------------------------------------------

# Semantic error type constants (mellea-specific categories)
ERROR_TYPE_RATE_LIMIT = "rate_limit"
ERROR_TYPE_TIMEOUT = "timeout"
ERROR_TYPE_CONTENT_POLICY = "content_policy"
ERROR_TYPE_AUTH = "auth"
ERROR_TYPE_INVALID_REQUEST = "invalid_request"
ERROR_TYPE_TRANSPORT_ERROR = "transport_error"
ERROR_TYPE_SERVER_ERROR = "server_error"
ERROR_TYPE_UNKNOWN = "unknown"


def classify_error(exc: BaseException) -> str:
    """Map an exception to a semantic error type string.

    Checks OpenAI SDK exception types first (when openai is installed), then
    falls back to stdlib exceptions and name-based heuristics.

    Args:
        exc: The exception to classify.

    Returns:
        One of the ``ERROR_TYPE_*`` constants.
    """
    # OpenAI SDK exceptions (optional dependency)
    try:
        import openai

        if isinstance(exc, openai.RateLimitError):
            return ERROR_TYPE_RATE_LIMIT
        if isinstance(exc, openai.APITimeoutError):
            return ERROR_TYPE_TIMEOUT
        if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)):
            return ERROR_TYPE_AUTH
        if isinstance(exc, openai.BadRequestError):
            # Content policy violations surface as BadRequestError with a specific code
            if getattr(exc, "code", None) == "content_policy_violation":
                return ERROR_TYPE_CONTENT_POLICY
            return ERROR_TYPE_INVALID_REQUEST
        if isinstance(exc, openai.APIConnectionError):
            return ERROR_TYPE_TRANSPORT_ERROR
        if isinstance(exc, openai.InternalServerError):
            return ERROR_TYPE_SERVER_ERROR
    except ImportError:
        pass

    # Stdlib exceptions
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        return ERROR_TYPE_TIMEOUT
    if isinstance(exc, ConnectionError):
        return ERROR_TYPE_TRANSPORT_ERROR

    # Name-based heuristics for provider-specific exceptions without explicit imports
    name_lower = type(exc).__name__.lower()
    if "ratelimit" in name_lower or "rate_limit" in name_lower:
        return ERROR_TYPE_RATE_LIMIT
    if "timeout" in name_lower:
        return ERROR_TYPE_TIMEOUT
    if "auth" in name_lower or "unauthorized" in name_lower:
        return ERROR_TYPE_AUTH
    if "content" in name_lower and "policy" in name_lower:
        return ERROR_TYPE_CONTENT_POLICY
    if (
        "connection" in name_lower
        or "network" in name_lower
        or "transport" in name_lower
    ):
        return ERROR_TYPE_TRANSPORT_ERROR
    if "server" in name_lower:
        return ERROR_TYPE_SERVER_ERROR

    return ERROR_TYPE_UNKNOWN


_error_counter: Any = None


def _get_error_counter() -> Any:
    """Get or create the LLM error counter (internal use only).

    Returns:
        Counter instrument for LLM errors.
    """
    global _error_counter

    if _error_counter is None:
        _error_counter = create_counter(
            "mellea.llm.errors",
            description="Total number of LLM errors categorized by semantic type",
            unit="{error}",
        )

    return _error_counter


def record_error(
    error_type: str, model: str, provider: str, exception_class: str
) -> None:
    """Record an LLM error metric.

    This is a no-op when metrics are disabled, ensuring zero overhead.

    Args:
        error_type: Semantic error category (use ``ERROR_TYPE_*`` constants).
        model: Model identifier (e.g. "gpt-4", "llama2:7b").
        provider: Provider name (e.g. "openai", "ollama").
        exception_class: Python exception class name (e.g. "RateLimitError").

    Example:
        record_error(
            error_type=ERROR_TYPE_RATE_LIMIT,
            model="gpt-4",
            provider="openai",
            exception_class="RateLimitError",
        )
    """
    if not _METRICS_ENABLED:
        return

    counter = _get_error_counter()
    counter.add(
        1,
        {
            "error_type": error_type,
            "gen_ai.request.model": model,
            "gen_ai.provider.name": provider,
            "error.type": exception_class,
        },
    )


# ---------------------------------------------------------------------------
# Cost counter
# ---------------------------------------------------------------------------

_cost_counter: Any = None


def _get_cost_counter() -> Any:
    """Get or create the LLM cost counter (internal use only).

    Returns:
        Counter instrument for LLM request cost.
    """
    global _cost_counter

    if _cost_counter is None:
        _cost_counter = create_counter(
            "mellea.llm.cost.usd",
            description="Estimated LLM request cost in USD",
            unit="USD",
        )

    return _cost_counter


def record_cost(cost: float, model: str, provider: str) -> None:
    """Record estimated LLM request cost in USD.

    This is a no-op when metrics are disabled, ensuring zero overhead.
    Only call this when pricing data is available (i.e., ``compute_cost`` returned
    a non-None value).

    Args:
        cost: Estimated request cost in US dollars.
        model: Model identifier (e.g. ``"gpt-4o"``, ``"claude-sonnet-4-6"``).
        provider: Provider name (e.g. ``"openai"``, ``"ollama"``).

    Example:
        record_cost(
            cost=0.0042,
            model="gpt-4o",
            provider="openai",
        )
    """
    if not _METRICS_ENABLED:
        return

    counter = _get_cost_counter()
    counter.add(cost, {"gen_ai.request.model": model, "gen_ai.provider.name": provider})


_sampling_attempts_counter: Any = None
_sampling_successes_counter: Any = None
_sampling_failures_counter: Any = None


def _get_sampling_attempts_counter() -> Any:
    """Get or create the sampling attempts counter (internal use only)."""
    global _sampling_attempts_counter

    if _sampling_attempts_counter is None:
        _sampling_attempts_counter = create_counter(
            "mellea.sampling.attempts",
            description="Total number of sampling attempts per strategy",
            unit="{attempt}",
        )
    return _sampling_attempts_counter


def _get_sampling_successes_counter() -> Any:
    """Get or create the sampling successes counter (internal use only)."""
    global _sampling_successes_counter

    if _sampling_successes_counter is None:
        _sampling_successes_counter = create_counter(
            "mellea.sampling.successes",
            description="Total number of successful sampling loops per strategy",
            unit="{sample}",
        )
    return _sampling_successes_counter


def _get_sampling_failures_counter() -> Any:
    """Get or create the sampling failures counter (internal use only)."""
    global _sampling_failures_counter

    if _sampling_failures_counter is None:
        _sampling_failures_counter = create_counter(
            "mellea.sampling.failures",
            description="Total number of failed sampling loops (budget exhausted) per strategy",
            unit="{failure}",
        )
    return _sampling_failures_counter


def record_sampling_attempt(strategy: str) -> None:
    """Record one sampling attempt for the given strategy.

    This is a no-op when metrics are disabled, ensuring zero overhead.

    Args:
        strategy: Sampling strategy class name (e.g. ``"RejectionSamplingStrategy"``).
    """
    if not _METRICS_ENABLED:
        return

    _get_sampling_attempts_counter().add(1, {"strategy": strategy})


def record_sampling_outcome(strategy: str, success: bool) -> None:
    """Record the final outcome (success or failure) of a sampling loop.

    This is a no-op when metrics are disabled, ensuring zero overhead.

    Args:
        strategy: Sampling strategy class name (e.g. ``"RejectionSamplingStrategy"``).
        success: ``True`` if at least one attempt passed all requirements.
    """
    if not _METRICS_ENABLED:
        return

    if success:
        _get_sampling_successes_counter().add(1, {"strategy": strategy})
    else:
        _get_sampling_failures_counter().add(1, {"strategy": strategy})


_requirement_checks_counter: Any = None
_requirement_failures_counter: Any = None


def _get_requirement_checks_counter() -> Any:
    """Get or create the requirement checks counter (internal use only)."""
    global _requirement_checks_counter

    if _requirement_checks_counter is None:
        _requirement_checks_counter = create_counter(
            "mellea.requirement.checks",
            description="Total number of requirement validation checks",
            unit="{check}",
        )
    return _requirement_checks_counter


def _get_requirement_failures_counter() -> Any:
    """Get or create the requirement failures counter (internal use only)."""
    global _requirement_failures_counter

    if _requirement_failures_counter is None:
        _requirement_failures_counter = create_counter(
            "mellea.requirement.failures",
            description="Total number of requirement validation failures",
            unit="{failure}",
        )
    return _requirement_failures_counter


def record_requirement_check(requirement: str) -> None:
    """Record one requirement validation check.

    This is a no-op when metrics are disabled, ensuring zero overhead.

    Args:
        requirement: Requirement class name (e.g. ``"LLMaJRequirement"``).
    """
    if not _METRICS_ENABLED:
        return

    _get_requirement_checks_counter().add(1, {"requirement": requirement})


def record_requirement_failure(requirement: str, reason: str) -> None:
    """Record one requirement validation failure.

    This is a no-op when metrics are disabled, ensuring zero overhead.

    Args:
        requirement: Requirement class name (e.g. ``"LLMaJRequirement"``).
        reason: Human-readable failure reason from ``ValidationResult.reason``.
    """
    if not _METRICS_ENABLED:
        return

    _get_requirement_failures_counter().add(
        1, {"requirement": requirement, "reason": reason}
    )


_tool_calls_counter: Any = None


def _get_tool_calls_counter() -> Any:
    """Get or create the tool calls counter (internal use only)."""
    global _tool_calls_counter

    if _tool_calls_counter is None:
        _tool_calls_counter = create_counter(
            "mellea.tool.calls",
            description="Total number of tool invocations by name and status",
            unit="{call}",
        )
    return _tool_calls_counter


def record_tool_call(tool: str, status: str) -> None:
    """Record one tool invocation.

    This is a no-op when metrics are disabled, ensuring zero overhead.

    Args:
        tool: Name of the tool that was invoked.
        status: ``"success"`` if the tool executed without error, ``"failure"`` otherwise.
    """
    if not _METRICS_ENABLED:
        return

    counter = _get_tool_calls_counter()
    counter.add(1, {"tool": tool, "status": status})


__all__ = [
    "classify_error",
    "create_counter",
    "create_histogram",
    "create_up_down_counter",
    "is_metrics_enabled",
    "record_cost",
    "record_error",
    "record_request_duration",
    "record_requirement_check",
    "record_requirement_failure",
    "record_sampling_attempt",
    "record_sampling_outcome",
    "record_token_usage_metrics",
    "record_tool_call",
    "record_ttfb",
]
