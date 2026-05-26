"""Tests for operational counter metrics (sampling, requirement, tool).

Integration tests use InMemoryMetricReader to verify counter values and attributes.
Unit tests verify no-op behaviour when metrics are disabled.
"""

import pytest

try:
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not OTEL_AVAILABLE, reason="OpenTelemetry not installed"
)


@pytest.fixture
def clean_metrics_env(monkeypatch):
    """Enable metrics and reset all module state for each test."""
    monkeypatch.setenv("MELLEA_METRICS_ENABLED", "true")
    monkeypatch.delenv("MELLEA_METRICS_CONSOLE", raising=False)

    import importlib

    import mellea.telemetry.metrics

    importlib.reload(mellea.telemetry.metrics)
    yield
    importlib.reload(mellea.telemetry.metrics)


def _setup_in_memory_provider(metrics_module):
    """Wire an InMemoryMetricReader into the metrics module globals."""
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    metrics_module._meter_provider = provider
    metrics_module._meter = provider.get_meter("mellea")
    # Reset all operational counter globals so they bind to the new meter
    metrics_module._sampling_attempts_counter = None
    metrics_module._sampling_successes_counter = None
    metrics_module._sampling_failures_counter = None
    metrics_module._requirement_checks_counter = None
    metrics_module._requirement_failures_counter = None
    metrics_module._tool_calls_counter = None
    return reader, provider


def _data_points_for(metrics_data, metric_name):
    """Return all data points for the named metric."""
    if metrics_data is None:
        return []
    data_points = []
    for rm in metrics_data.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == metric_name:
                    data_points.extend(metric.data.data_points)
    return data_points


# ---------------------------------------------------------------------------
# Sampling — attempts
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_record_sampling_attempt_basic(clean_metrics_env):
    """Sampling attempt counter records correct value and strategy attribute."""
    from mellea.telemetry import metrics as m

    reader, provider = _setup_in_memory_provider(m)

    m.record_sampling_attempt("RejectionSamplingStrategy")

    provider.force_flush()
    dps = _data_points_for(reader.get_metrics_data(), "mellea.sampling.attempts")

    assert len(dps) == 1
    assert dps[0].value == 1
    assert dict(dps[0].attributes)["strategy"] == "RejectionSamplingStrategy"


@pytest.mark.integration
def test_record_sampling_attempt_accumulation(clean_metrics_env):
    """Multiple attempts accumulate correctly."""
    from mellea.telemetry import metrics as m

    reader, provider = _setup_in_memory_provider(m)

    for _ in range(3):
        m.record_sampling_attempt("RejectionSamplingStrategy")

    provider.force_flush()
    dps = _data_points_for(reader.get_metrics_data(), "mellea.sampling.attempts")

    assert len(dps) == 1
    assert dps[0].value == 3


@pytest.mark.integration
def test_record_sampling_attempt_multiple_strategies(clean_metrics_env):
    """Different strategies are tracked as separate attribute sets."""
    from mellea.telemetry import metrics as m

    reader, provider = _setup_in_memory_provider(m)

    m.record_sampling_attempt("RejectionSamplingStrategy")
    m.record_sampling_attempt("MultiTurnStrategy")

    provider.force_flush()
    dps = _data_points_for(reader.get_metrics_data(), "mellea.sampling.attempts")

    assert len(dps) == 2
    strategies = {dict(dp.attributes)["strategy"] for dp in dps}
    assert strategies == {"RejectionSamplingStrategy", "MultiTurnStrategy"}


# ---------------------------------------------------------------------------
# Sampling — outcomes
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_record_sampling_outcome_success(clean_metrics_env):
    """Success outcome increments the successes counter."""
    from mellea.telemetry import metrics as m

    reader, provider = _setup_in_memory_provider(m)

    m.record_sampling_outcome("RejectionSamplingStrategy", success=True)

    provider.force_flush()
    success_dps = _data_points_for(
        reader.get_metrics_data(), "mellea.sampling.successes"
    )
    failure_dps = _data_points_for(
        reader.get_metrics_data(), "mellea.sampling.failures"
    )

    assert len(success_dps) == 1
    assert success_dps[0].value == 1
    assert len(failure_dps) == 0


@pytest.mark.integration
def test_record_sampling_outcome_failure(clean_metrics_env):
    """Failure outcome increments the failures counter."""
    from mellea.telemetry import metrics as m

    reader, provider = _setup_in_memory_provider(m)

    m.record_sampling_outcome("RejectionSamplingStrategy", success=False)

    provider.force_flush()
    success_dps = _data_points_for(
        reader.get_metrics_data(), "mellea.sampling.successes"
    )
    failure_dps = _data_points_for(
        reader.get_metrics_data(), "mellea.sampling.failures"
    )

    assert len(success_dps) == 0
    assert len(failure_dps) == 1
    assert failure_dps[0].value == 1


# ---------------------------------------------------------------------------
# Requirement checks
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_record_requirement_check_basic(clean_metrics_env):
    """Requirement check counter records correct value and requirement attribute."""
    from mellea.telemetry import metrics as m

    reader, provider = _setup_in_memory_provider(m)

    m.record_requirement_check("LLMaJRequirement")

    provider.force_flush()
    dps = _data_points_for(reader.get_metrics_data(), "mellea.requirement.checks")

    assert len(dps) == 1
    assert dps[0].value == 1
    assert dict(dps[0].attributes)["requirement"] == "LLMaJRequirement"


@pytest.mark.integration
def test_record_requirement_check_multiple_types(clean_metrics_env):
    """Different requirement types are tracked separately."""
    from mellea.telemetry import metrics as m

    reader, provider = _setup_in_memory_provider(m)

    m.record_requirement_check("LLMaJRequirement")
    m.record_requirement_check("PythonExecutionReq")

    provider.force_flush()
    dps = _data_points_for(reader.get_metrics_data(), "mellea.requirement.checks")

    assert len(dps) == 2
    req_names = {dict(dp.attributes)["requirement"] for dp in dps}
    assert req_names == {"LLMaJRequirement", "PythonExecutionReq"}


@pytest.mark.integration
def test_record_requirement_failure_attributes(clean_metrics_env):
    """Requirement failure counter records requirement and reason attributes."""
    from mellea.telemetry import metrics as m

    reader, provider = _setup_in_memory_provider(m)

    m.record_requirement_failure(
        "LLMaJRequirement", "Output did not satisfy constraint"
    )

    provider.force_flush()
    dps = _data_points_for(reader.get_metrics_data(), "mellea.requirement.failures")

    assert len(dps) == 1
    attrs = dict(dps[0].attributes)
    assert attrs["requirement"] == "LLMaJRequirement"
    assert attrs["reason"] == "Output did not satisfy constraint"


@pytest.mark.integration
def test_record_requirement_failure_accumulation(clean_metrics_env):
    """Multiple failures with the same attributes accumulate correctly."""
    from mellea.telemetry import metrics as m

    reader, provider = _setup_in_memory_provider(m)

    m.record_requirement_failure("LLMaJRequirement", "unknown")
    m.record_requirement_failure("LLMaJRequirement", "unknown")

    provider.force_flush()
    dps = _data_points_for(reader.get_metrics_data(), "mellea.requirement.failures")

    assert len(dps) == 1
    assert dps[0].value == 2


# ---------------------------------------------------------------------------
# Tool calls
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_record_tool_call_success(clean_metrics_env):
    """Tool call counter records name and success status."""
    from mellea.telemetry import metrics as m

    reader, provider = _setup_in_memory_provider(m)

    m.record_tool_call("search", "success")

    provider.force_flush()
    dps = _data_points_for(reader.get_metrics_data(), "mellea.tool.calls")

    assert len(dps) == 1
    attrs = dict(dps[0].attributes)
    assert attrs["tool"] == "search"
    assert attrs["status"] == "success"


@pytest.mark.integration
def test_record_tool_call_failure(clean_metrics_env):
    """Tool call counter records failure status separately from success."""
    from mellea.telemetry import metrics as m

    reader, provider = _setup_in_memory_provider(m)

    m.record_tool_call("search", "success")
    m.record_tool_call("search", "failure")

    provider.force_flush()
    dps = _data_points_for(reader.get_metrics_data(), "mellea.tool.calls")

    assert len(dps) == 2
    statuses = {dict(dp.attributes)["status"] for dp in dps}
    assert statuses == {"success", "failure"}


@pytest.mark.integration
def test_record_tool_call_multiple_tools(clean_metrics_env):
    """Different tool names are tracked as separate attribute sets."""
    from mellea.telemetry import metrics as m

    reader, provider = _setup_in_memory_provider(m)

    m.record_tool_call("search", "success")
    m.record_tool_call("calculator", "success")

    provider.force_flush()
    dps = _data_points_for(reader.get_metrics_data(), "mellea.tool.calls")

    assert len(dps) == 2
    tools = {dict(dp.attributes)["tool"] for dp in dps}
    assert tools == {"search", "calculator"}


# ---------------------------------------------------------------------------
# Unit: no-op when metrics disabled
# ---------------------------------------------------------------------------


def test_record_sampling_attempt_noop_when_disabled(monkeypatch):
    """record_sampling_attempt is a no-op when metrics are disabled."""
    import importlib

    import mellea.telemetry.metrics as m

    importlib.reload(m)
    monkeypatch.setattr(m, "_METRICS_ENABLED", False)

    # Should not raise and should not create any counter
    m.record_sampling_attempt("RejectionSamplingStrategy")
    assert m._sampling_attempts_counter is None


def test_record_sampling_outcome_noop_when_disabled(monkeypatch):
    """record_sampling_outcome is a no-op when metrics are disabled."""
    import importlib

    import mellea.telemetry.metrics as m

    importlib.reload(m)
    monkeypatch.setattr(m, "_METRICS_ENABLED", False)

    m.record_sampling_outcome("RejectionSamplingStrategy", success=True)
    assert m._sampling_successes_counter is None


def test_record_requirement_check_noop_when_disabled(monkeypatch):
    """record_requirement_check is a no-op when metrics are disabled."""
    import importlib

    import mellea.telemetry.metrics as m

    importlib.reload(m)
    monkeypatch.setattr(m, "_METRICS_ENABLED", False)

    m.record_requirement_check("LLMaJRequirement")
    assert m._requirement_checks_counter is None


def test_record_requirement_failure_noop_when_disabled(monkeypatch):
    """record_requirement_failure is a no-op when metrics are disabled."""
    import importlib

    import mellea.telemetry.metrics as m

    importlib.reload(m)
    monkeypatch.setattr(m, "_METRICS_ENABLED", False)

    m.record_requirement_failure("LLMaJRequirement", "reason")
    assert m._requirement_failures_counter is None


def test_record_tool_call_noop_when_disabled(monkeypatch):
    """record_tool_call is a no-op when metrics are disabled."""
    import importlib

    import mellea.telemetry.metrics as m

    importlib.reload(m)
    monkeypatch.setattr(m, "_METRICS_ENABLED", False)

    m.record_tool_call("search", "success")
    assert m._tool_calls_counter is None
