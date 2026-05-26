"""Unit tests for tracing helper functions — no OpenTelemetry installation required.

_set_attribute_safe and end_backend_span operate on any object with a
set_attribute / end method, so these tests use MagicMock spans and run
unconditionally. test_set_span_error_records_exception calls into the real
OTel trace API and is skipped when opentelemetry is not installed.
"""

from unittest.mock import MagicMock, patch

import pytest

from mellea.telemetry.tracing import (
    _set_attribute_safe,
    end_backend_span,
    set_span_error,
)

# --- _set_attribute_safe type-conversion ---


def test_set_attribute_safe_none_value_no_op():
    span = MagicMock()
    _set_attribute_safe(span, "key", None)
    span.set_attribute.assert_not_called()


def test_set_attribute_safe_bool():
    span = MagicMock()
    _set_attribute_safe(span, "flag", True)
    span.set_attribute.assert_called_once_with("flag", True)


def test_set_attribute_safe_int():
    span = MagicMock()
    _set_attribute_safe(span, "count", 42)
    span.set_attribute.assert_called_once_with("count", 42)


def test_set_attribute_safe_str():
    span = MagicMock()
    _set_attribute_safe(span, "name", "hello")
    span.set_attribute.assert_called_once_with("name", "hello")


def test_set_attribute_safe_list_converted_to_string_list():
    span = MagicMock()
    _set_attribute_safe(span, "items", [1, 2, 3])
    span.set_attribute.assert_called_once_with("items", ["1", "2", "3"])


def test_set_attribute_safe_unsupported_type_stringified():
    span = MagicMock()
    _set_attribute_safe(span, "obj", {"nested": "dict"})
    span.set_attribute.assert_called_once()
    call_args = span.set_attribute.call_args
    assert call_args.args[0] == "obj"
    assert isinstance(call_args.args[1], str)


# --- set_span_error — requires opentelemetry for trace.Status ---


def test_set_span_error_records_exception():
    pytest.importorskip(
        "opentelemetry",
        reason="opentelemetry not installed — install mellea[telemetry]",
    )
    span = MagicMock()
    exc = ValueError("something went wrong")

    with patch("mellea.telemetry.tracing._OTEL_AVAILABLE", True):
        set_span_error(span, exc)

    span.record_exception.assert_called_once_with(exc)
    span.set_status.assert_called_once()


# --- end_backend_span ---


def test_end_backend_span_calls_end_on_span():
    span = MagicMock()
    end_backend_span(span)
    span.end.assert_called_once()


def test_end_backend_span_none_no_op():
    end_backend_span(None)
