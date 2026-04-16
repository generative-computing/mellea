"""Integration tests verifying MelleaLogger works correctly inside plugin hook dispatch.

Key properties verified:
- MelleaLogger.get_logger() is callable and usable from inside a hook handler.
- Log records emitted from inside a hook are captured.
- log_context fields set inside a hook appear on records from that hook.
- log_context fields set in the caller ARE visible inside AUDIT hook execution
  (AUDIT hooks are awaited in the same asyncio task, so ContextVar state is inherited).
"""

# pytest: integration

from __future__ import annotations

import logging
from typing import Any

import pytest

pytestmark = pytest.mark.integration

pytest.importorskip("cpex.framework")

import datetime

# ---------------------------------------------------------------------------
# Minimal mock backend (avoids real LLM calls)
# ---------------------------------------------------------------------------
from unittest.mock import MagicMock

from mellea.core.backend import Backend
from mellea.core.base import (
    CBlock,
    Context,
    GenerateLog,
    GenerateType,
    ModelOutputThunk,
)
from mellea.core.utils import (
    MelleaLogger,
    clear_log_context,
    log_context,
    set_log_context,
)
from mellea.plugins import PluginMode, hook, register
from mellea.plugins.manager import shutdown_plugins
from mellea.stdlib.context import SimpleContext


class _MockBackend(Backend):
    model_id = "mock-model"

    def __init__(self, *args, **kwargs):
        pass

    async def _generate_from_context(self, action, ctx, **kwargs):
        mot = MagicMock(spec=ModelOutputThunk)
        glog = GenerateLog()
        glog.prompt = "mocked prompt"
        mot._generate_log = glog
        mot.parsed_repr = None
        mot._start = datetime.datetime.now()

        async def _avalue():
            return "mocked output"

        mot.avalue = _avalue
        mot.value = "mocked output"
        return mot, SimpleContext()

    async def generate_from_raw(self, actions, ctx, **kwargs):
        return []


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
async def reset_plugins():
    """Shut down and reset the plugin manager after every test."""
    yield
    await shutdown_plugins()


@pytest.fixture(autouse=True)
def reset_log_context():
    """Ensure log context is clean before and after each test."""
    clear_log_context()
    yield
    clear_log_context()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMelleaLoggerInHooks:
    async def test_mellea_logger_callable_from_hook(self, caplog) -> None:
        """MelleaLogger.get_logger() is usable inside a hook handler without error."""
        from mellea.stdlib.components import Instruction
        from mellea.stdlib.sampling.base import RejectionSamplingStrategy

        fired: list[bool] = []

        @hook("sampling_loop_start", mode=PluginMode.AUDIT)
        async def log_hook(payload: Any, ctx: Any) -> None:
            logger = MelleaLogger.get_logger()
            logger.info("hook fired from MelleaLogger")
            fired.append(True)

        register(log_hook)

        with caplog.at_level(logging.INFO, logger="fancy_logger"):
            await RejectionSamplingStrategy(loop_budget=1).sample(
                Instruction("test"),
                context=SimpleContext(),
                backend=_MockBackend(),
                requirements=[],
                format=None,
                model_options=None,
                tool_calls=False,
                show_progress=False,
            )

        assert fired, "Hook did not fire"
        assert any("hook fired from MelleaLogger" in r.message for r in caplog.records)

    async def test_log_context_set_inside_hook_appears_on_hook_records(
        self, caplog
    ) -> None:
        """Context fields set inside a hook appear on records emitted in that hook."""
        from mellea.stdlib.components import Instruction
        from mellea.stdlib.sampling.base import RejectionSamplingStrategy

        hook_records: list[logging.LogRecord] = []

        @hook("sampling_loop_start", mode=PluginMode.AUDIT)
        async def context_hook(payload: Any, ctx: Any) -> None:
            with log_context(hook_trace_id="hook-abc"):
                logger = MelleaLogger.get_logger()
                # Emit via a plain handler so we can capture the LogRecord
                record = logger.makeRecord(
                    name="fancy_logger",
                    level=logging.INFO,
                    fn="test",
                    lno=0,
                    msg="inside hook",
                    args=(),
                    exc_info=None,
                )
                # Apply the context filter manually (as the logger would)
                from mellea.core.utils import ContextFilter

                ContextFilter().filter(record)
                hook_records.append(record)

        register(context_hook)

        await RejectionSamplingStrategy(loop_budget=1).sample(
            Instruction("test"),
            context=SimpleContext(),
            backend=_MockBackend(),
            requirements=[],
            format=None,
            model_options=None,
            tool_calls=False,
            show_progress=False,
        )

        assert hook_records, "Hook did not produce records"
        assert getattr(hook_records[0], "hook_trace_id", None) == "hook-abc"

    async def test_log_context_is_visible_inside_hook(self) -> None:
        """ContextVar state set in the caller IS visible inside hook execution.

        AUDIT hooks are awaited in the same asyncio task as the caller, so they
        inherit the caller's ContextVar copy. This is the documented behaviour:
        log_context fields set around a strategy.sample() call will appear on
        records emitted inside hook handlers too.
        """
        from mellea.stdlib.components import Instruction
        from mellea.stdlib.sampling.base import RejectionSamplingStrategy

        hook_saw_context: list[bool] = []

        set_log_context(outer_field="visible-in-hook")

        @hook("sampling_loop_start", mode=PluginMode.AUDIT)
        async def visibility_hook(payload: Any, ctx: Any) -> None:
            from mellea.core.utils import _log_context

            fields = _log_context.get()
            hook_saw_context.append("outer_field" in fields)

        register(visibility_hook)

        await RejectionSamplingStrategy(loop_budget=1).sample(
            Instruction("test"),
            context=SimpleContext(),
            backend=_MockBackend(),
            requirements=[],
            format=None,
            model_options=None,
            tool_calls=False,
            show_progress=False,
        )

        assert hook_saw_context, "Hook did not fire"
        assert hook_saw_context[0], (
            "log_context fields should be visible inside AUDIT hooks (same asyncio task)"
        )

    async def test_sampling_log_context_fields_present_on_success_record(
        self, caplog
    ) -> None:
        """strategy and loop_budget context fields appear on the SUCCESS log record."""
        from mellea.stdlib.components import Instruction
        from mellea.stdlib.sampling.base import RejectionSamplingStrategy

        with caplog.at_level(logging.INFO, logger="fancy_logger"):
            await RejectionSamplingStrategy(loop_budget=2).sample(
                Instruction("test"),
                context=SimpleContext(),
                backend=_MockBackend(),
                requirements=[],
                format=None,
                model_options=None,
                tool_calls=False,
                show_progress=False,
            )

        success_records = [r for r in caplog.records if r.getMessage() == "SUCCESS"]
        assert success_records, "No SUCCESS record found"
        record = success_records[0]
        assert getattr(record, "strategy", None) == "RejectionSamplingStrategy"
        assert getattr(record, "loop_budget", None) == 2
