"""Unit tests for MelleaLogger, JsonFormatter, ContextFilter, and OtelTraceFilter."""

# pytest: unit

import asyncio
import io
import json
import logging
import threading
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit

from mellea.core.utils import (
    RESERVED_LOG_RECORD_ATTRS,
    ContextFilter,
    CustomFormatter,
    JsonFormatter,
    MelleaLogger,
    OtelTraceFilter,
    clear_log_context,
    log_context,
    set_log_context,
)


@contextmanager
def _otel_span(trace_id: int, span_id: int, is_valid: bool = True):
    """Context manager that patches mellea.core.utils with a mock OTel span."""
    mock_ctx = MagicMock()
    mock_ctx.is_valid = is_valid
    mock_ctx.trace_id = trace_id
    mock_ctx.span_id = span_id
    mock_span = MagicMock()
    mock_span.get_span_context.return_value = mock_ctx
    mock_otel = MagicMock()
    mock_otel.get_current_span.return_value = mock_span
    with (
        patch("mellea.core.utils._OTEL_AVAILABLE", True),
        patch("mellea.core.utils.otel_trace", mock_otel, create=True),
    ):
        yield


def _make_record(msg: str = "hello", level: int = logging.INFO) -> logging.LogRecord:
    """Return a minimal LogRecord for use in formatter/filter tests."""
    record = logging.LogRecord(
        name="test",
        level=level,
        pathname="test_utils_logging.py",
        lineno=1,
        msg=msg,
        args=(),
        exc_info=None,
    )
    return record


# ---------------------------------------------------------------------------
# OtelTraceFilter
# ---------------------------------------------------------------------------


class TestOtelTraceFilter:
    def test_always_returns_true_without_otel(self):
        """Filter never suppresses records when OTel is unavailable."""
        with patch("mellea.core.utils._OTEL_AVAILABLE", False):
            f = OtelTraceFilter()
            record = _make_record()
            assert f.filter(record) is True

    def test_no_attributes_added_without_otel(self):
        """No trace_id/span_id are added to the record when OTel is unavailable."""
        with patch("mellea.core.utils._OTEL_AVAILABLE", False):
            f = OtelTraceFilter()
            record = _make_record()
            f.filter(record)
            assert not hasattr(record, "trace_id")
            assert not hasattr(record, "span_id")

    def test_always_returns_true_with_active_span(self):
        """Filter never suppresses records when OTel is available and span is active."""
        with _otel_span(0xABCD1234ABCD1234ABCD1234ABCD1234, 0x1234567890ABCDEF):
            f = OtelTraceFilter()
            record = _make_record()
            assert f.filter(record) is True

    def test_injects_trace_and_span_id_when_span_active(self):
        """trace_id and span_id are hex-formatted and added to the record."""
        with _otel_span(0xABCD1234ABCD1234ABCD1234ABCD1234, 0x1234567890ABCDEF):
            f = OtelTraceFilter()
            record = _make_record()
            f.filter(record)

        assert record.trace_id == "abcd1234abcd1234abcd1234abcd1234"
        assert len(record.trace_id) == 32
        assert record.span_id == "1234567890abcdef"
        assert len(record.span_id) == 16

    def test_no_attributes_when_span_invalid(self):
        """No trace_id/span_id are added when the current span context is invalid."""
        with _otel_span(0, 0, is_valid=False):
            f = OtelTraceFilter()
            record = _make_record()
            f.filter(record)

        assert not hasattr(record, "trace_id")
        assert not hasattr(record, "span_id")

    def test_always_returns_true_when_span_invalid(self):
        """Filter never suppresses records even when no active span context."""
        with _otel_span(0, 0, is_valid=False):
            f = OtelTraceFilter()
            record = _make_record()
            assert f.filter(record) is True


# ---------------------------------------------------------------------------
# JsonFormatter
# ---------------------------------------------------------------------------


class TestJsonFormatterCoreSchema:
    def test_returns_valid_json_string(self) -> None:
        fmt = JsonFormatter()
        output = fmt.format(_make_record("hi"))
        parsed = json.loads(output)
        assert isinstance(parsed, dict)

    def test_all_default_fields_present(self) -> None:
        fmt = JsonFormatter()
        parsed = json.loads(fmt.format(_make_record("hi")))
        for field in JsonFormatter._DEFAULT_FIELDS:
            assert field in parsed, f"missing field: {field}"

    def test_message_content(self) -> None:
        fmt = JsonFormatter()
        parsed = json.loads(fmt.format(_make_record("test message")))
        assert parsed["message"] == "test message"

    def test_level_name(self) -> None:
        fmt = JsonFormatter()
        parsed = json.loads(fmt.format(_make_record(level=logging.WARNING)))
        assert parsed["level"] == "WARNING"

    def test_exception_field_added_when_exc_info(self) -> None:
        fmt = JsonFormatter()
        try:
            raise ValueError("boom")
        except ValueError:
            import sys

            record = _make_record("oops")
            record.exc_info = sys.exc_info()
            parsed = json.loads(fmt.format(record))
        assert "exception" in parsed
        assert "ValueError" in parsed["exception"]

    def test_trace_fields_absent_without_filter(self):
        """trace_id and span_id are not in output when filter hasn't run."""
        fmt = JsonFormatter()
        record = _make_record()
        result = json.loads(fmt.format(record))

        assert "trace_id" not in result
        assert "span_id" not in result

    def test_trace_fields_present_when_populated(self):
        """trace_id and span_id appear in output when set on the record."""
        fmt = JsonFormatter()
        record = _make_record()
        record.trace_id = "abcd1234abcd1234abcd1234abcd1234"
        record.span_id = "1234567890abcdef"

        result = json.loads(fmt.format(record))

        assert result["trace_id"] == "abcd1234abcd1234abcd1234abcd1234"
        assert result["span_id"] == "1234567890abcdef"

    def test_trace_id_without_span_id_does_not_raise(self):
        """trace_id present but span_id absent must not raise AttributeError."""
        fmt = JsonFormatter()
        record = _make_record()
        record.trace_id = "abcd1234abcd1234abcd1234abcd1234"
        # span_id intentionally absent

        result = json.loads(fmt.format(record))

        assert result["trace_id"] == "abcd1234abcd1234abcd1234abcd1234"
        assert "span_id" not in result


class TestJsonFormatterFieldConfig:
    def test_include_fields_limits_output(self) -> None:
        fmt = JsonFormatter(include_fields=["timestamp", "level", "message"])
        parsed = json.loads(fmt.format(_make_record()))
        assert set(parsed.keys()) == {"timestamp", "level", "message"}

    def test_exclude_fields_removes_keys(self) -> None:
        fmt = JsonFormatter(exclude_fields=["process_id", "thread_id"])
        parsed = json.loads(fmt.format(_make_record()))
        assert "process_id" not in parsed
        assert "thread_id" not in parsed
        assert "level" in parsed  # other fields still present

    def test_extra_fields_merged(self) -> None:
        fmt = JsonFormatter(extra_fields={"service": "mellea", "env": "test"})
        parsed = json.loads(fmt.format(_make_record()))
        assert parsed["service"] == "mellea"
        assert parsed["env"] == "test"

    def test_extra_fields_override_core(self) -> None:
        # static extras come *after* core fields — they win on collision
        fmt = JsonFormatter(extra_fields={"level": "OVERRIDDEN"})
        parsed = json.loads(fmt.format(_make_record(level=logging.DEBUG)))
        assert parsed["level"] == "OVERRIDDEN"

    def test_timestamp_format_respected(self) -> None:
        fmt = JsonFormatter(timestamp_format="%Y")
        parsed = json.loads(fmt.format(_make_record()))
        # Should be just a 4-digit year
        assert len(parsed["timestamp"]) == 4
        assert parsed["timestamp"].isdigit()


class TestJsonFormatterContextInjection:
    def setup_method(self) -> None:
        clear_log_context()

    def teardown_method(self) -> None:
        clear_log_context()

    def test_context_fields_appear_in_output(self) -> None:
        set_log_context(request_id="abc-123")
        fmt = JsonFormatter()
        parsed = json.loads(fmt.format(_make_record()))
        assert parsed.get("request_id") == "abc-123"

    def test_multiple_context_fields(self) -> None:
        set_log_context(custom_trace="t1", request_id="r1", user="alice")
        fmt = JsonFormatter()
        parsed = json.loads(fmt.format(_make_record()))
        assert parsed["custom_trace"] == "t1"
        assert parsed["request_id"] == "r1"
        assert parsed["user"] == "alice"

    def test_clear_context_removes_fields(self) -> None:
        set_log_context(custom_trace="gone")
        clear_log_context()
        fmt = JsonFormatter()
        parsed = json.loads(fmt.format(_make_record()))
        assert "custom_trace" not in parsed

    def test_context_is_thread_local(self) -> None:
        """Fields set in one thread must not bleed into another."""
        results: dict[str, Any] = {}
        barrier = threading.Barrier(2)

        def worker_a() -> None:
            set_log_context(custom_trace="thread-a")
            barrier.wait()  # both threads read context at the same time
            fmt = JsonFormatter()
            results["a"] = json.loads(fmt.format(_make_record()))
            clear_log_context()

        def worker_b() -> None:
            # does NOT call set_log_context
            barrier.wait()  # both threads read context at the same time
            fmt = JsonFormatter()
            results["b"] = json.loads(fmt.format(_make_record()))

        ta = threading.Thread(target=worker_a)
        tb = threading.Thread(target=worker_b)
        ta.start()
        tb.start()
        ta.join()
        tb.join()

        assert results["a"].get("custom_trace") == "thread-a"
        assert "custom_trace" not in results["b"]


class TestContextFilter:
    def setup_method(self) -> None:
        clear_log_context()

    def teardown_method(self) -> None:
        clear_log_context()

    def test_filter_always_returns_true(self) -> None:
        f = ContextFilter()
        assert f.filter(_make_record()) is True

    def test_filter_attaches_context_fields_to_record(self) -> None:
        set_log_context(custom_span="span-999")
        f = ContextFilter()
        record = _make_record()
        f.filter(record)
        assert getattr(record, "custom_span", None) == "span-999"

    def test_filter_noop_when_no_context(self) -> None:
        f = ContextFilter()
        record = _make_record()
        f.filter(record)
        assert not hasattr(record, "trace_id")


@pytest.mark.unit
class TestMelleaLoggerLogLevel:
    def _reset(self) -> None:
        MelleaLogger.logger = None
        logging.getLogger("fancy_logger").handlers.clear()
        logging.getLogger("fancy_logger").setLevel(logging.NOTSET)

    def teardown_method(self) -> None:
        self._reset()

    def test_default_level_is_info(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MELLEA_LOG_LEVEL", raising=False)
        assert MelleaLogger._resolve_log_level() == logging.INFO

    def test_mellea_log_level_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MELLEA_LOG_LEVEL", "DEBUG")
        assert MelleaLogger._resolve_log_level() == logging.DEBUG

    def test_mellea_log_level_warning(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MELLEA_LOG_LEVEL", "WARNING")
        assert MelleaLogger._resolve_log_level() == logging.WARNING

    def test_invalid_level_falls_back_to_info(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MELLEA_LOG_LEVEL", "BOGUS")
        assert MelleaLogger._resolve_log_level() == logging.INFO


@pytest.mark.unit
class TestMelleaLoggerJsonConsole:
    def _reset(self) -> None:
        MelleaLogger.logger = None
        logger = logging.getLogger("fancy_logger")
        logger.handlers.clear()
        logger.setLevel(logging.NOTSET)

    def setup_method(self) -> None:
        self._reset()

    def teardown_method(self) -> None:
        self._reset()

    def _get_stream_handler(self) -> logging.StreamHandler:  # type: ignore[type-arg]
        logger = MelleaLogger.get_logger()
        handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler)]
        # RESTHandler is a subclass of Handler but not StreamHandler, so this
        # correctly picks the console handler.
        return handlers[0]

    def test_default_uses_custom_formatter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MELLEA_LOG_JSON", raising=False)
        handler = self._get_stream_handler()
        assert isinstance(handler.formatter, CustomFormatter)

    def test_json_console_enabled_with_true(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MELLEA_LOG_JSON", "true")
        handler = self._get_stream_handler()
        assert isinstance(handler.formatter, JsonFormatter)

    def test_json_console_enabled_with_1(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MELLEA_LOG_JSON", "1")
        handler = self._get_stream_handler()
        assert isinstance(handler.formatter, JsonFormatter)

    def test_json_console_enabled_with_yes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MELLEA_LOG_JSON", "yes")
        handler = self._get_stream_handler()
        assert isinstance(handler.formatter, JsonFormatter)

    def test_json_console_disabled_with_false(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MELLEA_LOG_JSON", "false")
        handler = self._get_stream_handler()
        assert isinstance(handler.formatter, CustomFormatter)


@pytest.mark.unit
class TestMelleaLoggerFiltersWired:
    def setup_method(self) -> None:
        MelleaLogger.logger = None
        logging.getLogger("fancy_logger").handlers.clear()
        logging.getLogger("fancy_logger").setLevel(logging.NOTSET)
        clear_log_context()

    def teardown_method(self) -> None:
        MelleaLogger.logger = None
        logging.getLogger("fancy_logger").handlers.clear()
        logging.getLogger("fancy_logger").setLevel(logging.NOTSET)
        clear_log_context()

    def test_context_filter_present(self) -> None:
        logger = MelleaLogger.get_logger()
        assert any(isinstance(f, ContextFilter) for f in logger.filters)

    def test_otel_filter_present(self) -> None:
        logger = MelleaLogger.get_logger()
        assert any(isinstance(f, OtelTraceFilter) for f in logger.filters)


class TestLogContext:
    def setup_method(self) -> None:
        clear_log_context()

    def teardown_method(self) -> None:
        clear_log_context()

    def test_fields_present_inside_block(self) -> None:
        fmt = JsonFormatter()
        with log_context(custom_trace="ctx-1"):
            parsed = json.loads(fmt.format(_make_record()))
            assert parsed["custom_trace"] == "ctx-1"

    def test_fields_removed_after_block(self) -> None:
        fmt = JsonFormatter()
        with log_context(custom_trace="ctx-2"):
            pass
        parsed = json.loads(fmt.format(_make_record()))
        assert "custom_trace" not in parsed

    def test_cleanup_on_exception(self) -> None:
        fmt = JsonFormatter()
        with pytest.raises(RuntimeError):
            with log_context(custom_trace="ctx-err"):
                raise RuntimeError("boom")
        parsed = json.loads(fmt.format(_make_record()))
        assert "custom_trace" not in parsed

    def test_nested_contexts_preserve_outer(self) -> None:
        fmt = JsonFormatter()
        with log_context(outer="yes"):
            with log_context(inner="yes"):
                parsed = json.loads(fmt.format(_make_record()))
                assert parsed["outer"] == "yes"
                assert parsed["inner"] == "yes"
            # inner should be gone, outer still present
            parsed = json.loads(fmt.format(_make_record()))
            assert parsed["outer"] == "yes"
            assert "inner" not in parsed
        # both gone
        parsed = json.loads(fmt.format(_make_record()))
        assert "outer" not in parsed

    def test_nested_same_key_restores_outer(self) -> None:
        fmt = JsonFormatter()
        with log_context(custom_trace="outer"):
            with log_context(custom_trace="inner"):
                parsed = json.loads(fmt.format(_make_record()))
                assert parsed["custom_trace"] == "inner"
            parsed = json.loads(fmt.format(_make_record()))
            assert parsed["custom_trace"] == "outer"
        parsed = json.loads(fmt.format(_make_record()))
        assert "custom_trace" not in parsed

    def test_rejects_reserved_attribute(self) -> None:
        with pytest.raises(ValueError, match="reserved"):
            with log_context(levelname="BAD"):
                pass


class TestLogContextAsyncIsolation:
    """Verify that concurrent asyncio tasks cannot contaminate each other's context."""

    def setup_method(self) -> None:
        clear_log_context()

    def teardown_method(self) -> None:
        clear_log_context()

    def test_concurrent_tasks_isolated(self) -> None:
        """Fields set inside one asyncio.Task must not bleed into a sibling task."""
        fmt = JsonFormatter()
        results: dict[str, Any] = {}

        async def task_a() -> None:
            with log_context(custom_trace="task-a"):
                # Yield so task_b can run and attempt to overwrite the context
                await asyncio.sleep(0)
                results["a"] = json.loads(fmt.format(_make_record()))

        async def task_b() -> None:
            with log_context(custom_trace="task-b"):
                await asyncio.sleep(0)
                results["b"] = json.loads(fmt.format(_make_record()))

        async def run() -> None:
            await asyncio.gather(
                asyncio.create_task(task_a()), asyncio.create_task(task_b())
            )

        asyncio.run(run())

        assert results["a"].get("custom_trace") == "task-a"
        assert results["b"].get("custom_trace") == "task-b"

    def test_task_context_does_not_leak_after_completion(self) -> None:
        """A task's context fields must not persist into the caller after the task ends."""
        fmt = JsonFormatter()

        async def child() -> None:
            set_log_context(custom_trace="child-task")

        async def run() -> dict[str, object]:
            await asyncio.create_task(child())
            # The caller's context should be unaffected
            return json.loads(fmt.format(_make_record()))

        parsed = asyncio.run(run())
        assert "custom_trace" not in parsed


class TestReservedAttributeValidation:
    def setup_method(self) -> None:
        clear_log_context()

    def teardown_method(self) -> None:
        clear_log_context()

    def test_set_log_context_rejects_reserved_key(self) -> None:
        with pytest.raises(ValueError, match="reserved"):
            set_log_context(module="bad")

    def test_set_log_context_rejects_multiple_reserved(self) -> None:
        with pytest.raises(ValueError, match="reserved"):
            set_log_context(thread="x", process="y")

    def test_set_log_context_accepts_non_reserved(self) -> None:
        set_log_context(custom_field="fine")
        fmt = JsonFormatter()
        parsed = json.loads(fmt.format(_make_record()))
        assert parsed["custom_field"] == "fine"

    def test_set_log_context_rejects_trace_id(self) -> None:
        with pytest.raises(ValueError, match="reserved"):
            set_log_context(trace_id="x")

    def test_set_log_context_rejects_span_id(self) -> None:
        with pytest.raises(ValueError, match="reserved"):
            set_log_context(span_id="x")

    def test_reserved_set_is_non_empty(self) -> None:
        assert len(RESERVED_LOG_RECORD_ATTRS) > 10


@pytest.mark.unit
class TestGetLoggerThreadSafety:
    def setup_method(self) -> None:
        MelleaLogger.logger = None
        logging.getLogger("fancy_logger").handlers.clear()
        logging.getLogger("fancy_logger").setLevel(logging.NOTSET)

    def teardown_method(self) -> None:
        MelleaLogger.logger = None
        logging.getLogger("fancy_logger").handlers.clear()
        logging.getLogger("fancy_logger").setLevel(logging.NOTSET)

    def test_concurrent_get_logger_returns_same_instance(self) -> None:
        """Multiple threads calling get_logger() must all get the same object."""
        results: list[logging.Logger] = []
        barrier = threading.Barrier(4)

        def worker() -> None:
            barrier.wait()
            results.append(MelleaLogger.get_logger())

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 4
        assert all(r is results[0] for r in results)


class TestJsonFormatterFormatSignature:
    def test_format_returns_str(self) -> None:
        """JsonFormatter.format returns str — no type: ignore needed."""
        fmt = JsonFormatter()
        result = fmt.format(_make_record("check"))
        assert isinstance(result, str)
        json.loads(result)  # also valid JSON


class TestFormatAsDict:
    def test_format_as_dict_returns_same_as_internal(self) -> None:
        fmt = JsonFormatter()
        record = _make_record("hi")
        assert fmt.format_as_dict(record) == fmt._build_log_dict(record)

    def test_format_as_dict_malformed_args_does_not_raise(self) -> None:
        fmt = JsonFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="x.py",
            lineno=1,
            msg="val: %s %s",
            args=("only_one",),
            exc_info=None,
        )
        result = fmt.format_as_dict(record)
        assert "message" in result
        assert "format error" in result["message"]


class TestIncludeFieldsValidation:
    def test_valid_include_fields_accepted(self) -> None:
        fmt = JsonFormatter(include_fields=["timestamp", "level", "message"])
        parsed = json.loads(fmt.format(_make_record()))
        assert set(parsed.keys()) == {"timestamp", "level", "message"}

    def test_unknown_include_field_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown field names"):
            JsonFormatter(include_fields=["timestamp", "bogus_field"])

    def test_all_default_fields_accepted(self) -> None:
        fmt = JsonFormatter(include_fields=list(JsonFormatter._DEFAULT_FIELDS))
        parsed = json.loads(fmt.format(_make_record()))
        assert set(parsed.keys()) == set(JsonFormatter._DEFAULT_FIELDS)


# ---------------------------------------------------------------------------
# CustomFormatter
# ---------------------------------------------------------------------------


class TestCustomFormatter:
    def test_returns_string(self):
        """format() always returns a string."""
        fmt = CustomFormatter(datefmt="%H:%M:%S")
        record = _make_record()
        result = fmt.format(record)
        assert isinstance(result, str)

    def test_no_trace_suffix_without_filter(self):
        """No [trace_id=… span_id=…] suffix when filter has not run."""
        fmt = CustomFormatter(datefmt="%H:%M:%S")
        record = _make_record()
        result = fmt.format(record)
        assert "trace_id=" not in result
        assert "span_id=" not in result

    def test_trace_suffix_appended_when_populated(self):
        """[trace_id=… span_id=…] suffix is appended when record has trace context."""
        fmt = CustomFormatter(datefmt="%H:%M:%S")
        record = _make_record()
        record.trace_id = "abcd1234abcd1234abcd1234abcd1234"
        record.span_id = "1234567890abcdef"

        result = fmt.format(record)

        assert (
            "[trace_id=abcd1234abcd1234abcd1234abcd1234 span_id=1234567890abcdef]"
            in result
        )

    def test_all_log_levels_format(self):
        """CustomFormatter handles all standard log levels without error."""
        fmt = CustomFormatter(datefmt="%H:%M:%S")
        for level in (
            logging.DEBUG,
            logging.INFO,
            logging.WARNING,
            logging.ERROR,
            logging.CRITICAL,
        ):
            logger = logging.getLogger("test")
            record = logger.makeRecord(
                name="test",
                level=level,
                fn="f.py",
                lno=1,
                msg="msg",
                args=(),
                exc_info=None,
            )
            result = fmt.format(record)
            assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Integration: filter → formatter round-trip
# ---------------------------------------------------------------------------


class TestFilterFormatterIntegration:
    def setup_method(self) -> None:
        MelleaLogger.logger = None
        logging.getLogger("fancy_logger").handlers.clear()
        logging.getLogger("fancy_logger").filters.clear()

    def teardown_method(self) -> None:
        MelleaLogger.logger = None
        logging.getLogger("fancy_logger").handlers.clear()
        logging.getLogger("fancy_logger").filters.clear()

    def test_json_formatter_picks_up_filter_output(self):
        """OtelTraceFilter + JsonFormatter round-trip injects trace context."""
        with _otel_span(0x00000000000000000000000000000001, 0x0000000000000002):
            f = OtelTraceFilter()
            fmt = JsonFormatter()
            record = _make_record("integration test")
            f.filter(record)
            result = json.loads(fmt.format(record))

        assert result["trace_id"] == "00000000000000000000000000000001"
        assert result["span_id"] == "0000000000000002"

    def test_custom_formatter_picks_up_filter_output(self):
        """OtelTraceFilter + CustomFormatter round-trip appends trace suffix."""
        with _otel_span(0x00000000000000000000000000000001, 0x0000000000000002):
            f = OtelTraceFilter()
            fmt = CustomFormatter(datefmt="%H:%M:%S")
            record = _make_record("integration test")
            f.filter(record)
            result = fmt.format(record)

        assert "trace_id=00000000000000000000000000000001" in result
        assert "span_id=0000000000000002" in result

    def test_logger_singleton_with_otel_filter_and_json_formatter(self):
        """MelleaLogger.get_logger() with OtelTraceFilter produces trace context in JSON output."""
        with _otel_span(0x00000000000000000000000000000003, 0x0000000000000004):
            logger = MelleaLogger.get_logger()
            # Capture output with JsonFormatter
            stream = io.StringIO()
            handler = logging.StreamHandler(stream)
            handler.setFormatter(JsonFormatter())
            logger.addHandler(handler)

            try:
                # Log a message - filters are applied automatically by the logger
                logger.info("logger integration test")

                # Parse the JSON output
                output = stream.getvalue().strip()
                result = json.loads(output)

                assert result["trace_id"] == "00000000000000000000000000000003"
                assert result["span_id"] == "0000000000000004"
                assert result["message"] == "logger integration test"
            finally:
                logger.removeHandler(handler)

    def test_logger_singleton_with_otel_filter_and_custom_formatter(self):
        """MelleaLogger.get_logger() with OtelTraceFilter produces trace context in custom format output."""
        with _otel_span(0x00000000000000000000000000000005, 0x0000000000000006):
            logger = MelleaLogger.get_logger()
            # Capture output with CustomFormatter
            stream = io.StringIO()
            handler = logging.StreamHandler(stream)
            handler.setFormatter(CustomFormatter(datefmt="%H:%M:%S"))
            logger.addHandler(handler)

            try:
                # Log a message - filters are applied automatically by the logger
                logger.info("logger integration test")

                # Check the formatted output
                output = stream.getvalue().strip()

                assert "trace_id=00000000000000000000000000000005" in output
                assert "span_id=0000000000000006" in output
                assert "logger integration test" in output
            finally:
                logger.removeHandler(handler)
