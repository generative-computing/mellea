# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""OTLP/HTTP selection and base endpoint paths for both signals."""

from unittest.mock import patch

import pytest

pytest.importorskip("opentelemetry")

from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
    OTLPMetricExporter as HTTPMetricExporter,
)
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter as HTTPSpanExporter,
)

from mellea.telemetry import metrics, tracing
from mellea.telemetry._otlp_config import otlp_protocol_endpoint


@pytest.mark.parametrize("signal", ["traces", "metrics"])
def test_signal_specific_http_endpoint_is_used_as_is(monkeypatch, signal):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
    monkeypatch.setenv(
        f"OTEL_EXPORTER_OTLP_{signal.upper()}_ENDPOINT",
        "http://collector.example/custom",
    )

    assert otlp_protocol_endpoint(signal) == (
        "http/protobuf",
        "http://collector.example/custom",
    )


def test_http_protocol_uses_trace_path_from_general_endpoint(monkeypatch):
    monkeypatch.setenv("MELLEA_TRACES_OTLP", "true")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)

    with patch.object(
        tracing, "HTTPOTLPSpanExporter", wraps=HTTPSpanExporter, create=True
    ) as exporter:
        provider = tracing._setup_tracer_provider()
        try:
            exporter.assert_called_once_with(endpoint="http://localhost:4318/v1/traces")
        finally:
            provider.shutdown()


def test_http_protocol_uses_metric_path_from_general_endpoint(monkeypatch):
    monkeypatch.setenv("MELLEA_METRICS_OTLP", "true")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", raising=False)

    with patch.object(
        metrics, "HTTPOTLPMetricExporter", wraps=HTTPMetricExporter, create=True
    ) as exporter:
        provider = metrics._setup_meter_provider()
        try:
            exporter.assert_called_once_with(
                endpoint="http://localhost:4318/v1/metrics"
            )
        finally:
            provider.shutdown()
