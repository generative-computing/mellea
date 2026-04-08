"""Unit tests for FancyLogger, JsonFormatter, and ContextFilter enhancements."""

import json
import logging
import threading
from typing import Any

import pytest

from mellea.core.utils import (
    _RESERVED_LOG_RECORD_ATTRS,
    ContextFilter,
    FancyLogger,
    JsonFormatter,
    clear_log_context,
    log_context,
    set_log_context,
)


def _make_record(msg: str = "hello", level: int = logging.INFO) -> logging.LogRecord:
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
        set_log_context(trace_id="abc-123")
        fmt = JsonFormatter()
        parsed = json.loads(fmt.format(_make_record()))
        assert parsed.get("trace_id") == "abc-123"

    def test_multiple_context_fields(self) -> None:
        set_log_context(trace_id="t1", request_id="r1", user="alice")
        fmt = JsonFormatter()
        parsed = json.loads(fmt.format(_make_record()))
        assert parsed["trace_id"] == "t1"
        assert parsed["request_id"] == "r1"
        assert parsed["user"] == "alice"

    def test_clear_context_removes_fields(self) -> None:
        set_log_context(trace_id="gone")
        clear_log_context()
        fmt = JsonFormatter()
        parsed = json.loads(fmt.format(_make_record()))
        assert "trace_id" not in parsed

    def test_context_is_thread_local(self) -> None:
        """Fields set in one thread must not bleed into another."""
        results: dict[str, Any] = {}

        def worker_a() -> None:
            set_log_context(trace_id="thread-a")
            import time

            time.sleep(0.05)
            fmt = JsonFormatter()
            results["a"] = json.loads(fmt.format(_make_record()))
            clear_log_context()

        def worker_b() -> None:
            # does NOT call set_log_context
            import time

            time.sleep(0.02)
            fmt = JsonFormatter()
            results["b"] = json.loads(fmt.format(_make_record()))

        ta = threading.Thread(target=worker_a)
        tb = threading.Thread(target=worker_b)
        ta.start()
        tb.start()
        ta.join()
        tb.join()

        assert results["a"].get("trace_id") == "thread-a"
        assert "trace_id" not in results["b"]


class TestContextFilter:
    def setup_method(self) -> None:
        clear_log_context()

    def teardown_method(self) -> None:
        clear_log_context()

    def test_filter_always_returns_true(self) -> None:
        f = ContextFilter()
        assert f.filter(_make_record()) is True

    def test_filter_attaches_context_fields_to_record(self) -> None:
        set_log_context(span_id="span-999")
        f = ContextFilter()
        record = _make_record()
        f.filter(record)
        assert getattr(record, "span_id", None) == "span-999"

    def test_filter_noop_when_no_context(self) -> None:
        f = ContextFilter()
        record = _make_record()
        f.filter(record)
        assert not hasattr(record, "trace_id")


class TestFancyLoggerLogLevel:
    def _reset(self) -> None:
        FancyLogger.logger = None
        logging.getLogger("fancy_logger").handlers.clear()
        logging.getLogger("fancy_logger").setLevel(logging.NOTSET)

    def teardown_method(self) -> None:
        self._reset()

    def test_default_level_is_info(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MELLEA_LOG_LEVEL", raising=False)
        monkeypatch.delenv("DEBUG", raising=False)
        assert FancyLogger._resolve_log_level() == logging.INFO

    def test_mellea_log_level_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MELLEA_LOG_LEVEL", "DEBUG")
        assert FancyLogger._resolve_log_level() == logging.DEBUG

    def test_mellea_log_level_warning(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MELLEA_LOG_LEVEL", "WARNING")
        assert FancyLogger._resolve_log_level() == logging.WARNING

    def test_legacy_debug_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MELLEA_LOG_LEVEL", raising=False)
        monkeypatch.setenv("DEBUG", "1")
        assert FancyLogger._resolve_log_level() == logging.DEBUG

    def test_mellea_log_level_takes_precedence_over_debug(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MELLEA_LOG_LEVEL", "WARNING")
        monkeypatch.setenv("DEBUG", "1")
        assert FancyLogger._resolve_log_level() == logging.WARNING

    def test_invalid_level_falls_back_to_info(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MELLEA_LOG_LEVEL", "BOGUS")
        monkeypatch.delenv("DEBUG", raising=False)
        assert FancyLogger._resolve_log_level() == logging.INFO


class TestFancyLoggerJsonConsole:
    def _reset(self) -> None:
        FancyLogger.logger = None
        logger = logging.getLogger("fancy_logger")
        logger.handlers.clear()
        logger.setLevel(logging.NOTSET)

    def setup_method(self) -> None:
        self._reset()

    def teardown_method(self) -> None:
        self._reset()

    def _get_stream_handler(self) -> logging.StreamHandler:  # type: ignore[type-arg]
        logger = FancyLogger.get_logger()
        handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler)]
        # RESTHandler is a subclass of Handler but not StreamHandler, so this
        # correctly picks the console handler.
        return handlers[0]

    def test_default_uses_custom_formatter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from mellea.core.utils import CustomFormatter

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
        from mellea.core.utils import CustomFormatter

        monkeypatch.setenv("MELLEA_LOG_JSON", "false")
        handler = self._get_stream_handler()
        assert isinstance(handler.formatter, CustomFormatter)


class TestFancyLoggerContextFilterWired:
    def setup_method(self) -> None:
        FancyLogger.logger = None
        logging.getLogger("fancy_logger").handlers.clear()
        logging.getLogger("fancy_logger").setLevel(logging.NOTSET)
        clear_log_context()

    def teardown_method(self) -> None:
        FancyLogger.logger = None
        logging.getLogger("fancy_logger").handlers.clear()
        logging.getLogger("fancy_logger").setLevel(logging.NOTSET)
        clear_log_context()

    def test_context_filter_present(self) -> None:
        logger = FancyLogger.get_logger()
        assert any(isinstance(f, ContextFilter) for f in logger.filters)


class TestLogContext:
    def setup_method(self) -> None:
        clear_log_context()

    def teardown_method(self) -> None:
        clear_log_context()

    def test_fields_present_inside_block(self) -> None:
        fmt = JsonFormatter()
        with log_context(trace_id="ctx-1"):
            parsed = json.loads(fmt.format(_make_record()))
            assert parsed["trace_id"] == "ctx-1"

    def test_fields_removed_after_block(self) -> None:
        fmt = JsonFormatter()
        with log_context(trace_id="ctx-2"):
            pass
        parsed = json.loads(fmt.format(_make_record()))
        assert "trace_id" not in parsed

    def test_cleanup_on_exception(self) -> None:
        fmt = JsonFormatter()
        with pytest.raises(RuntimeError):
            with log_context(trace_id="ctx-err"):
                raise RuntimeError("boom")
        parsed = json.loads(fmt.format(_make_record()))
        assert "trace_id" not in parsed

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
        with log_context(trace_id="outer"):
            with log_context(trace_id="inner"):
                parsed = json.loads(fmt.format(_make_record()))
                assert parsed["trace_id"] == "inner"
            parsed = json.loads(fmt.format(_make_record()))
            assert parsed["trace_id"] == "outer"
        parsed = json.loads(fmt.format(_make_record()))
        assert "trace_id" not in parsed

    def test_rejects_reserved_attribute(self) -> None:
        with pytest.raises(ValueError, match="reserved"):
            with log_context(levelname="BAD"):
                pass


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

    def test_reserved_set_is_non_empty(self) -> None:
        assert len(_RESERVED_LOG_RECORD_ATTRS) > 10


class TestGetLoggerThreadSafety:
    def setup_method(self) -> None:
        FancyLogger.logger = None
        logging.getLogger("fancy_logger").handlers.clear()
        logging.getLogger("fancy_logger").setLevel(logging.NOTSET)

    def teardown_method(self) -> None:
        FancyLogger.logger = None
        logging.getLogger("fancy_logger").handlers.clear()
        logging.getLogger("fancy_logger").setLevel(logging.NOTSET)

    def test_concurrent_get_logger_returns_same_instance(self) -> None:
        """Multiple threads calling get_logger() must all get the same object."""
        results: list[logging.Logger] = []
        barrier = threading.Barrier(4)

        def worker() -> None:
            barrier.wait()
            results.append(FancyLogger.get_logger())

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
