# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests: the token-id retention path balances its telemetry.

The retention path (OpenAI backend) must MATERIALIZE the reply to derive the ids it
retains, so the thunk it returns is ALREADY computed. A computed thunk short-circuits
`astream()`, which is where a normal turn's `generation_post_call` fires -- yet
`generation_pre_call` already fired in the wrapper and opened a span. The fix defers the
post-call to `Backend.generate_from_context`, which fires it AFTER assigning
`generation_id`.

The unit tests (`test_openai_token_id_postcall_unit.py`, `test_hook_call_sites.py`) prove
the flag and the fire against RECORDING hooks. These drive the same shape through the REAL
`BackendTracingPlugin` and `LatencyMetricsPlugin` with in-memory OTel exporters, proving
the observable outcomes those hooks are supposed to produce:

    1. The span opened by pre_call is actually CLOSED -- the exact regression, since the
       old in-path fire carried `generation_id=None`, which `on_post_call` drops, so the
       span leaked in `_in_flight_spans` forever.
    2. The duration metric records a NON-NEGATIVE value -- the old raw-completion path
       never stamped `_gen.start`, so latency came through as -1 ms (-0.001 s).

A `MagicMock`-free flagged backend stands in for the OpenAI backend so no server or
`/tokenize` route is needed; the telemetry plumbing under test is entirely real.
"""

import datetime

import pytest

pytest.importorskip(
    "opentelemetry", reason="opentelemetry not installed — install mellea[telemetry]"
)
pytest.importorskip("cpex", reason="cpex not installed — install mellea[hooks]")

from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from mellea.core.backend import Backend
from mellea.core.base import GenerateLog, ModelOutputThunk
from mellea.plugins.manager import (
    disable_background_collection,
    discard_background_tasks,
    drain_background_tasks,
    enable_background_collection,
)
from mellea.stdlib.context import SimpleContext
from mellea.telemetry import tracing
from test.telemetry.conftest import reset_metrics_state, reset_tracing_state

pytestmark = [pytest.mark.integration]


class _FlaggedComputedBackend(Backend):
    """Minimal backend mirroring the token-id retention path's thunk contract.

    Returns an already-computed thunk with `_gen.start` stamped (so latency is real)
    and `fire_post_call_on_return` set (so the wrapper fires the deferred post_call).
    No LLM, no server: only the telemetry pairing is under test.
    """

    def __init__(self, retention: dict[str, int] | None = None) -> None:
        self._model_id = "mock-model"
        self._provider = "mock-provider"
        self.retention = retention

    async def _generate_from_context(self, action, ctx, **kwargs):
        mot = ModelOutputThunk(value="materialized reply")
        glog = GenerateLog()
        glog.prompt = "pre-tokenized prompt"
        mot._generate_log = glog
        mot._gen.start = datetime.datetime.now()
        mot._call.fire_post_call_on_return = True
        mot.generation.model = self._model_id
        mot.generation.provider = self._provider
        if self.retention is not None:
            mot.generation.token_id_retention = dict(self.retention)
        return mot, SimpleContext()

    async def _generate_from_raw(self, actions, ctx, **kwargs):  # pragma: no cover
        raise NotImplementedError


@pytest.fixture
def enabled_tracing(monkeypatch):
    monkeypatch.setenv("MELLEA_TRACES_ENABLED", "true")
    reset_tracing_state()
    yield
    reset_tracing_state()


@pytest.fixture
def span_exporter(enabled_tracing):
    """Attach an in-memory span exporter to the active tracer provider."""
    if tracing._tracer_provider is None:
        pytest.skip("Telemetry not initialized")
    exporter = InMemorySpanExporter()
    tracing._tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    yield exporter
    exporter.clear()


@pytest.fixture
def metric_reader(monkeypatch):
    """Wire an InMemoryMetricReader into the metrics module and register plugins.

    `LatencyMetricsPlugin` records on a FIRE_AND_FORGET hook, so background collection
    is enabled here; the test must `await drain_background_tasks()` before reading.
    """
    from mellea.telemetry import metrics as metrics_module

    enable_background_collection()
    discard_background_tasks()
    monkeypatch.setenv("MELLEA_METRICS_ENABLED", "true")
    monkeypatch.delenv("MELLEA_METRICS_CONSOLE", raising=False)
    reset_metrics_state()
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    metrics_module._meter_provider = provider
    metrics_module._meter = provider.get_meter("mellea")
    metrics_module._duration_histogram = None
    metrics_module._ttfb_histogram = None
    metrics_module._time_per_output_chunk_histogram = None
    metrics_module._token_id_retention_counter = None
    metrics_module._token_id_reused_prompt_tokens_histogram = None
    yield reader
    monkeypatch.setenv("MELLEA_METRICS_ENABLED", "false")
    reset_metrics_state()
    disable_background_collection()


def _duration_points(reader: InMemoryMetricReader):
    """All data points for the request-duration histogram."""
    data = reader.get_metrics_data()
    points: list = []
    if data is None:
        return points
    for rm in data.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == "mellea.llm.request.duration":
                    points.extend(metric.data.data_points)
    return points


def _metric_points(reader: InMemoryMetricReader, name: str):
    """All data points for the named instrument."""
    data = reader.get_metrics_data()
    points: list = []
    if data is None:
        return points
    for rm in data.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == name:
                    points.extend(metric.data.data_points)
    return points


async def test_deferred_post_call_closes_the_generation_span(span_exporter) -> None:
    """The pre_call span is ended, not leaked, once the wrapper fires the deferred post.

    `_in_flight_spans` keys the open span by `generation_id`; `on_post_call` closes it by
    the same id. The regression this guards: the old in-path fire passed
    `generation_id=None`, which `on_post_call` drops, so the entry stayed forever.
    """
    backend = _FlaggedComputedBackend()
    await backend.generate_from_context(ModelOutputThunk("hi"), SimpleContext())

    # No span left open: the deferred post_call reached the plugin with a real id.
    assert tracing._in_flight_spans == {}
    # And the span was actually recorded (ended) by the exporter.
    tracing._tracer_provider.force_flush()
    finished = span_exporter.get_finished_spans()
    assert len(finished) == 1
    assert finished[0].name == "chat mock-model"


async def test_deferred_post_call_records_non_negative_duration(metric_reader) -> None:
    """The retained turn's latency metric is >= 0, not the old -0.001 s.

    `_gen.start` is stamped on this path, so `_elapsed_ms()` is real; the wrapper passes
    it to the post_call payload, which `LatencyMetricsPlugin` divides into seconds.
    """
    backend = _FlaggedComputedBackend()
    await backend.generate_from_context(ModelOutputThunk("hi"), SimpleContext())
    await drain_background_tasks()

    points = _duration_points(metric_reader)
    assert len(points) == 1, "the retained turn recorded no duration metric"
    assert points[0].sum >= 0.0, f"negative duration recorded: {points[0].sum}"


# --- Retention signal: the client-side reuse counts reach the exporters -------


async def test_retention_signal_reaches_the_metrics_exporter(metric_reader) -> None:
    """A retaining turn records its reused-prefix size, tagged `reused=true`.

    This is the client-side signal a caller without access to the server's cache
    counters relies on: it says what Mellea sent, not that the server's cache hit.
    """
    backend = _FlaggedComputedBackend(
        retention={
            "reused_prompt_tokens": 12,
            "new_prompt_tokens": 3,
            "prompt_tokens": 15,
        }
    )
    await backend.generate_from_context(ModelOutputThunk("hi"), SimpleContext())
    await drain_background_tasks()

    turns = _metric_points(metric_reader, "mellea.token_id_retention.turns")
    assert len(turns) == 1, "the retaining turn recorded no retention metric"
    assert turns[0].attributes["reused"] == "true"
    assert turns[0].attributes["model"] == "mock-model"
    assert turns[0].attributes["provider"] == "mock-provider"

    reused = _metric_points(
        metric_reader, "mellea.token_id_retention.reused_prompt_tokens"
    )
    assert len(reused) == 1
    assert reused[0].sum == 12


async def test_no_reuse_is_recorded_as_a_turn_with_reused_false(metric_reader) -> None:
    """A turn sent as ids that reused nothing still counts, tagged `reused=false`.

    A first turn and a re-baselined turn both land here. Recording them keeps the
    denominator honest: reuse rate is derivable from this counter alone.
    """
    backend = _FlaggedComputedBackend(
        retention={
            "reused_prompt_tokens": 0,
            "new_prompt_tokens": 9,
            "prompt_tokens": 9,
        }
    )
    await backend.generate_from_context(ModelOutputThunk("hi"), SimpleContext())
    await drain_background_tasks()

    turns = _metric_points(metric_reader, "mellea.token_id_retention.turns")
    assert len(turns) == 1
    assert turns[0].attributes["reused"] == "false"


async def test_non_retaining_turn_records_no_retention_metric(metric_reader) -> None:
    """A turn that did not go through the id transport records nothing at all.

    Keeps the instrument's cardinality bounded by actual use of the feature, and keeps
    `reused=false` meaning "sent as ids, reused none" rather than "not applicable".
    """
    backend = _FlaggedComputedBackend()
    await backend.generate_from_context(ModelOutputThunk("hi"), SimpleContext())
    await drain_background_tasks()

    assert _metric_points(metric_reader, "mellea.token_id_retention.turns") == []


async def test_retention_signal_reaches_the_generation_span(span_exporter) -> None:
    """The reuse counts land on the generation span as `mellea.token_id_retention.*`.

    Deliberately not named `cache_hit`: the span reports what the client sent. The
    attribute's absence on a non-retaining turn is what makes its presence meaningful.
    """
    backend = _FlaggedComputedBackend(
        retention={
            "reused_prompt_tokens": 12,
            "new_prompt_tokens": 3,
            "prompt_tokens": 15,
        }
    )
    await backend.generate_from_context(ModelOutputThunk("hi"), SimpleContext())

    tracing._tracer_provider.force_flush()
    finished = span_exporter.get_finished_spans()
    assert len(finished) == 1
    attrs = finished[0].attributes
    assert attrs["mellea.token_id_retention.reused_prompt_tokens"] == 12
    assert attrs["mellea.token_id_retention.new_prompt_tokens"] == 3
    assert attrs["mellea.token_id_retention.prompt_tokens"] == 15


async def test_non_retaining_turn_leaves_the_span_attributes_off(span_exporter) -> None:
    """No retention attributes on a turn that did not send ids."""
    backend = _FlaggedComputedBackend()
    await backend.generate_from_context(ModelOutputThunk("hi"), SimpleContext())

    tracing._tracer_provider.force_flush()
    finished = span_exporter.get_finished_spans()
    assert len(finished) == 1
    assert not [
        k for k in finished[0].attributes if k.startswith("mellea.token_id_retention.")
    ]
