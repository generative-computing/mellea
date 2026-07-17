# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for OpenTelemetry tracing public API."""

import pytest

pytest.importorskip(
    "opentelemetry", reason="opentelemetry not installed — install mellea[telemetry]"
)

from mellea.telemetry import is_content_tracing_enabled, is_tracing_enabled, tracing
from mellea.telemetry.tracing import get_backend_tracer
from test.telemetry.conftest import reset_tracing_state


@pytest.fixture
def enable_tracing(monkeypatch):
    """Enable tracing for the duration of a test."""
    monkeypatch.setenv("MELLEA_TRACES_ENABLED", "true")
    reset_tracing_state()
    yield
    reset_tracing_state()


@pytest.fixture
def disable_tracing(monkeypatch):
    """Ensure tracing is disabled for the duration of a test."""
    monkeypatch.delenv("MELLEA_TRACES_ENABLED", raising=False)
    monkeypatch.delenv("MELLEA_TRACES_OTLP", raising=False)
    monkeypatch.delenv("MELLEA_TRACES_CONSOLE", raising=False)
    reset_tracing_state()
    yield
    reset_tracing_state()


def test_telemetry_disabled_by_default(disable_tracing):
    """Test that telemetry is disabled by default."""
    assert not is_tracing_enabled()


def test_tracing_enabled(enable_tracing):
    """Test that tracing can be enabled."""
    assert is_tracing_enabled()


@pytest.mark.parametrize(
    "env, expected",
    [
        ({}, False),
        ({"MELLEA_TRACES_CONTENT": "true"}, True),
        ({"OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "true"}, True),
    ],
    ids=["disabled-by-default", "mellea-var", "otel-standard-var"],
)
def test_content_tracing(monkeypatch, env, expected):
    """Content tracing honors both MELLEA_TRACES_CONTENT and the OTel standard var."""
    monkeypatch.delenv("MELLEA_TRACES_CONTENT", raising=False)
    monkeypatch.delenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", raising=False
    )
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    reset_tracing_state()

    assert is_content_tracing_enabled() is expected


def test_otlp_traces_endpoint_honored(monkeypatch):
    """OTEL_EXPORTER_OTLP_TRACES_ENDPOINT should activate the OTLP exporter."""
    monkeypatch.setenv("MELLEA_TRACES_ENABLED", "true")
    monkeypatch.setenv("MELLEA_TRACES_OTLP", "true")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://localhost:4317")
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    reset_tracing_state()

    assert get_backend_tracer() is not None
    reset_tracing_state()


def test_otlp_warns_when_no_endpoint_configured(monkeypatch, recwarn):
    """MELLEA_TRACES_OTLP=true with no endpoint set must warn."""
    monkeypatch.setenv("MELLEA_TRACES_OTLP", "true")
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)

    tracing._setup_tracer_provider()

    no_endpoint_warnings = [
        w for w in recwarn.list if "no endpoint is configured" in str(w.message)
    ]
    assert len(no_endpoint_warnings) == 1


def test_otlp_falls_back_to_generic_endpoint(monkeypatch, recwarn):
    """OTEL_EXPORTER_OTLP_ENDPOINT should be picked up when the traces-specific var is unset."""
    monkeypatch.setenv("MELLEA_TRACES_OTLP", "true")
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")

    tracing._setup_tracer_provider()

    no_endpoint_warnings = [
        w for w in recwarn.list if "no endpoint is configured" in str(w.message)
    ]
    assert no_endpoint_warnings == []
