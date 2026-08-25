# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests verifying that hooks fire at actual Mellea call sites.

Each test registers a hook recorder, triggers the actual code path (Backend,
functional.py, sampling/base.py, session.py), and asserts that the hook fired
with the expected payload shape.

All tests use lightweight mock backends so no real LLM API calls are made.
"""

from __future__ import annotations

import asyncio
import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.integration

pytest.importorskip("cpex.framework")

from mellea.core.backend import Backend
from mellea.core.base import (
    CBlock,
    Component,
    ComputedModelOutputThunk,
    Context,
    GenerateLog,
    GenerateType,
    ModelOutputThunk,
)
from mellea.core.requirement import Requirement, ValidationResult
from mellea.core.sampling import SamplingResult
from mellea.plugins import HookType, PluginResult, hook, register
from mellea.stdlib.components import Instruction, Message
from mellea.stdlib.context import ChatContext, SimpleContext
from mellea.stdlib.sampling.base import RejectionSamplingStrategy
from mellea.stdlib.sampling.budget_forcing import BudgetForcingSamplingStrategy
from mellea.stdlib.sampling.sofai import SOFAISamplingStrategy

# ---------------------------------------------------------------------------
# Mock backend (module-level so it can be used as a class in session tests)
# ---------------------------------------------------------------------------


_MOCK_RAW_USAGE = {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}


def _make_raw_mot(value: str, usage: dict[str, Any] | None = None) -> ModelOutputThunk:
    """Build a real, computed MOT for raw-path tests."""
    mot = ModelOutputThunk(value=value)
    mot.generation.usage = usage
    mot.generation.model = "mock-model"
    mot.generation.provider = "mock-provider"
    return mot


class _MockBackend(Backend):
    """Minimal backend that returns a faked ModelOutputThunk — no LLM API calls."""

    model_id = "mock-model"

    def __init__(self, *args, **kwargs):
        # Accept but discard constructor arguments; real backends need model_id etc.
        self._model_id: str = "mock-model"
        self._provider: str = "mock-provider"

    async def _generate_from_context(self, action, ctx, **kwargs):
        mot = ModelOutputThunk(value="mocked output string")
        glog = GenerateLog()
        glog.prompt = "mocked formatted prompt"
        mot._generate_log = glog
        mot._gen.start = datetime.datetime.now()
        # Return a new context of the same type to mimic real context evolution.
        # ChatContext is needed so SOFAI's repair() assertion passes.
        new_ctx = ChatContext() if isinstance(ctx, ChatContext) else SimpleContext()
        return mot, new_ctx

    async def _generate_from_raw(self, actions, ctx, **kwargs):
        results = [_make_raw_mot(f"mocked raw output for {a}") for a in actions]
        return results, _MOCK_RAW_USAGE


async def _noop_process(mot, chunk):
    if mot._underlying_value is None:
        mot._underlying_value = ""
    mot._underlying_value += str(chunk)


async def _noop_post_process(mot):
    return


def _make_thunk():
    mot = ModelOutputThunk(value=None)
    mot._gen.generate_type = GenerateType.ASYNC
    mot._gen.process = _noop_process
    mot._gen.post_process = _noop_post_process
    mot._call.action = CBlock("test")
    mot._gen.chunk_size = 0
    mot._gen.start = datetime.datetime.now()
    return mot


# ---------------------------------------------------------------------------
# Generation hook call sites
# ---------------------------------------------------------------------------


class TestGenerationHookCallSites:
    """GENERATION_PRE_CALL and GENERATION_POST_CALL fire in Backend.generate_from_context()."""

    async def test_generation_pre_call_fires_once(self) -> None:
        """GENERATION_PRE_CALL fires exactly once per generate_from_context() call."""
        observed: list[Any] = []

        @hook("generation_pre_call")
        async def recorder(payload: Any, ctx: Any) -> Any:
            observed.append(payload)
            return None

        register(recorder)
        backend = _MockBackend()
        action = CBlock("hello world")
        await backend.generate_from_context(action, MagicMock(spec=Context))

        assert len(observed) == 1

    async def test_generation_pre_call_payload_has_action_and_context(self) -> None:
        """GENERATION_PRE_CALL payload carries the action CBlock and the context."""
        observed: list[Any] = []

        @hook("generation_pre_call")
        async def recorder(payload: Any, ctx: Any) -> Any:
            observed.append(payload)
            return None

        register(recorder)
        backend = _MockBackend()
        action = CBlock("specific input text")
        mock_ctx = MagicMock(spec=Context)
        await backend.generate_from_context(action, mock_ctx)

        p = observed[0]

        assert isinstance(p.action, CBlock)
        assert p.action.value == action.value
        assert p.context is not None

    async def test_generation_post_call_fires_once(self) -> None:
        """GENERATION_POST_CALL fires exactly once after generate_from_context() returns."""
        observed: list[Any] = []

        @hook("generation_post_call")
        async def recorder(payload: Any, ctx: Any) -> Any:
            observed.append(payload)
            return None

        register(recorder)

        mot = _make_thunk()
        await mot._gen.queue.put("hello")
        await mot._gen.queue.put("goodbye")
        await mot._gen.queue.put(None)  # sentinel for being done

        await mot.avalue()
        assert len(observed) == 1

    async def test_generation_post_call_model_output_is_the_returned_thunk(
        self,
    ) -> None:
        """GENERATION_POST_CALL payload.model_output IS the ModelOutputThunk returned."""
        observed: list[Any] = []

        @hook("generation_post_call")
        async def recorder(payload: Any, ctx: Any) -> Any:
            observed.append(payload)
            return None

        register(recorder)

        mot = _make_thunk()
        await mot._gen.queue.put("hello")
        await mot._gen.queue.put("goodbye")
        await mot._gen.queue.put(None)  # sentinel for being done
        await mot.avalue()

        assert observed[0].model_output is not None

    async def test_generation_post_call_latency_ms_is_non_negative(self) -> None:
        """GENERATION_POST_CALL payload.latency_ms >= 0."""
        observed: list[Any] = []

        @hook("generation_post_call")
        async def recorder(payload: Any, ctx: Any) -> Any:
            observed.append(payload)
            return None

        register(recorder)

        mot = _make_thunk()
        await mot._gen.queue.put("hello")
        await mot._gen.queue.put("goodbye")
        await mot._gen.queue.put(None)  # sentinel for being done

        await asyncio.sleep(1)
        await mot.avalue()

        assert observed[0].latency_ms >= 0

    async def test_generation_pre_call_mutation_is_applied_before_generation(
        self,
    ) -> None:
        """GENERATION_PRE_CALL mutations reach the backend generation call."""
        order: list[str] = []
        captured_kwargs: dict[str, Any] = {}

        class RecordingBackend(_MockBackend):
            async def _generate_from_context(self, action, ctx, **kwargs):
                order.append("generate")
                captured_kwargs.update(kwargs)
                return await super()._generate_from_context(action, ctx, **kwargs)

        async def fake_invoke_hook(hook_type, payload, **_kwargs):
            assert hook_type is HookType.GENERATION_PRE_CALL
            order.append("hook")
            modified = payload.model_copy(
                update={"model_options": {"temperature": 0.25}, "tool_calls": True}
            )
            return (
                PluginResult(continue_processing=True, modified_payload=modified),
                modified,
            )

        backend = RecordingBackend()

        with (
            patch("mellea.core.backend.has_plugins", return_value=True),
            patch("mellea.core.backend.invoke_hook", side_effect=fake_invoke_hook),
        ):
            await backend.generate_from_context(
                CBlock("hook order"),
                MagicMock(spec=Context),
                model_options={"temperature": 1.0},
                tool_calls=False,
            )

        assert order == ["hook", "generate"]
        assert captured_kwargs["model_options"] == {"temperature": 0.25}
        assert captured_kwargs["tool_calls"] is True

    async def test_generation_pre_call_mutation_propagates_with_real_plugin(
        self,
    ) -> None:
        """GENERATION_PRE_CALL mutations survive the cpex policy layer."""
        captured_kwargs: dict[str, Any] = {}

        class RecordingBackend(_MockBackend):
            async def _generate_from_context(self, action, ctx, **kwargs):
                captured_kwargs.update(kwargs)
                return await super()._generate_from_context(action, ctx, **kwargs)

        @hook("generation_pre_call")
        async def mutator(payload: Any, ctx: Any) -> Any:
            return PluginResult(
                continue_processing=True,
                modified_payload=payload.model_copy(
                    update={
                        "model_options": {"temperature": 0.25},
                        "format": dict,
                        "tool_calls": True,
                    }
                ),
            )

        register(mutator)
        backend = RecordingBackend()
        await backend.generate_from_context(
            CBlock("hook order"),
            MagicMock(spec=Context),
            model_options={"temperature": 1.0},
            tool_calls=False,
        )

        assert captured_kwargs["model_options"] == {"temperature": 0.25}
        assert captured_kwargs["format"] is dict
        assert captured_kwargs["tool_calls"] is True

    async def test_generation_id_on_pre_call_payload_is_uuid(self) -> None:
        """Backend.generate_from_context generates a UUID and puts it on pre_call."""
        import uuid

        observed: list[Any] = []

        @hook("generation_pre_call")
        async def recorder(payload: Any, ctx: Any) -> Any:
            observed.append(payload)
            return None

        register(recorder)
        backend = _MockBackend()
        await backend.generate_from_context(CBlock("hi"), MagicMock(spec=Context))

        gen_id = observed[0].generation_id
        assert gen_id
        uuid.UUID(gen_id)  # Raises if not a valid UUID

    async def test_generation_id_stashed_on_returned_mot(self) -> None:
        """The same generation_id is stashed on the returned ModelOutputThunk."""
        observed: list[Any] = []

        @hook("generation_pre_call")
        async def recorder(payload: Any, ctx: Any) -> Any:
            observed.append(payload)
            return None

        register(recorder)
        backend = _MockBackend()
        mot, _ = await backend.generate_from_context(
            CBlock("hi"), MagicMock(spec=Context)
        )

        assert mot._call.generation_id == observed[0].generation_id

    async def test_generation_post_call_carries_generation_id(self) -> None:
        """astream() fires post_call with the MOT's _generation_id."""
        observed: list[Any] = []

        @hook("generation_post_call")
        async def recorder(payload: Any, ctx: Any) -> Any:
            observed.append(payload)
            return None

        register(recorder)

        mot = _make_thunk()
        mot._call.generation_id = "gid-post-1"
        await mot._gen.queue.put("hello")
        await mot._gen.queue.put(None)
        await mot.avalue()

        assert observed[0].generation_id == "gid-post-1"

    async def test_generation_error_fires_with_payload(self) -> None:
        """GENERATION_ERROR fires with the original exception and generation_id."""
        observed: list[Any] = []

        @hook("generation_error")
        async def recorder(payload: Any, ctx: Any) -> Any:
            observed.append(payload)
            return None

        register(recorder)

        mot = _make_thunk()
        mot._call.generation_id = "gid-err-1"
        error = ConnectionError("server unavailable")
        await mot._gen.queue.put(error)

        with pytest.raises(ConnectionError, match="server unavailable"):
            await mot.astream()

        assert len(observed) == 1
        assert observed[0].exception is error
        assert observed[0].model_output is not None
        assert observed[0].generation_id == "gid-err-1"

    async def test_cancel_generation_fires_error_with_supplied_exception(self) -> None:
        """cancel_generation(error=...) fires GENERATION_ERROR with that exception."""
        observed: list[Any] = []

        @hook("generation_error")
        async def recorder(payload: Any, ctx: Any) -> Any:
            observed.append(payload)
            return None

        register(recorder)

        mot = ModelOutputThunk(value=None)
        mot._call.generation_id = "gid-cancel-1"
        cause = ValueError("validator rejected")

        await mot.cancel_generation(error=cause)

        assert len(observed) == 1
        assert observed[0].exception is cause
        assert observed[0].model_output is not None
        assert observed[0].generation_id == "gid-cancel-1"
        assert mot._cancelled is True

    async def test_cancel_generation_fires_error_with_default_runtimeerror(
        self,
    ) -> None:
        """cancel_generation() with no error fires GENERATION_ERROR with a generic RuntimeError."""
        observed: list[Any] = []

        @hook("generation_error")
        async def recorder(payload: Any, ctx: Any) -> Any:
            observed.append(payload)
            return None

        register(recorder)

        mot = ModelOutputThunk(value=None)
        mot._call.generation_id = "gid-cancel-2"

        await mot.cancel_generation()

        assert len(observed) == 1
        assert isinstance(observed[0].exception, RuntimeError)
        assert "cancelled" in str(observed[0].exception).lower()
        assert observed[0].generation_id == "gid-cancel-2"

    async def test_generation_error_fires_on_sync_raise_in_generate(self) -> None:
        """A synchronous raise inside `_generate_from_context` still fires GENERATION_ERROR."""
        observed_pre: list[Any] = []
        observed_err: list[Any] = []

        @hook("generation_pre_call")
        async def pre_recorder(payload: Any, ctx: Any) -> Any:
            observed_pre.append(payload)
            return None

        @hook("generation_error")
        async def err_recorder(payload: Any, ctx: Any) -> Any:
            observed_err.append(payload)
            return None

        register(pre_recorder)
        register(err_recorder)

        class _RaisingBackend(_MockBackend):
            async def _generate_from_context(self, action, ctx, **kwargs):
                raise RuntimeError("setup failure")

        backend = _RaisingBackend()
        with pytest.raises(RuntimeError, match="setup failure"):
            await backend.generate_from_context(CBlock("hi"), MagicMock(spec=Context))

        assert len(observed_pre) == 1
        assert len(observed_err) == 1
        assert observed_err[0].generation_id == observed_pre[0].generation_id
        assert isinstance(observed_err[0].exception, RuntimeError)
        assert observed_err[0].model_output is None


# ---------------------------------------------------------------------------
# Generation batch hook call sites
# ---------------------------------------------------------------------------


class TestGenerationBatchHookCallSites:
    """GENERATION_BATCH_PRE/POST/ERROR fire in Backend.generate_from_raw()."""

    async def test_batch_pre_call_fires_once(self) -> None:
        """GENERATION_BATCH_PRE_CALL fires exactly once per generate_from_raw() call."""
        observed: list[Any] = []

        @hook("generation_batch_pre_call")
        async def recorder(payload: Any, ctx: Any) -> Any:
            observed.append(payload)
            return None

        register(recorder)
        backend = _MockBackend()
        actions: list[Component[Any] | CBlock] = [CBlock("a"), CBlock("b")]
        await backend.generate_from_raw(actions, MagicMock(spec=Context))

        assert len(observed) == 1

    async def test_batch_pre_call_payload_carries_actions_and_metadata(self) -> None:
        """Pre-call payload carries actions, num_actions, model, provider, generation_id."""
        observed: list[Any] = []

        @hook("generation_batch_pre_call")
        async def recorder(payload: Any, ctx: Any) -> Any:
            observed.append(payload)
            return None

        register(recorder)
        backend = _MockBackend()
        actions: list[Component[Any] | CBlock] = [CBlock("first"), CBlock("second")]
        await backend.generate_from_raw(actions, MagicMock(spec=Context))

        p = observed[0]
        assert len(p.actions) == 2
        assert p.num_actions == 2
        assert p.model == "mock-model"
        assert p.provider == "mock-provider"
        assert p.generation_id is not None

    async def test_batch_generation_id_on_pre_call_payload_is_uuid(self) -> None:
        """Backend.generate_from_raw generates a UUID and puts it on pre_call."""
        import uuid

        observed: list[Any] = []

        @hook("generation_batch_pre_call")
        async def recorder(payload: Any, ctx: Any) -> Any:
            observed.append(payload)
            return None

        register(recorder)
        backend = _MockBackend()
        await backend.generate_from_raw([CBlock("a")], MagicMock(spec=Context))

        gen_id = observed[0].generation_id
        assert gen_id
        uuid.UUID(gen_id)  # Raises if not a valid UUID

    async def test_batch_pre_call_mutation_propagates(self) -> None:
        """GENERATION_BATCH_PRE_CALL mutations reach the backend generation call."""
        order: list[str] = []
        captured_kwargs: dict[str, Any] = {}

        class RecordingBackend(_MockBackend):
            async def _generate_from_raw(self, actions, ctx, **kwargs):
                order.append("generate")
                captured_kwargs.update(kwargs)
                return await super()._generate_from_raw(actions, ctx, **kwargs)

        async def fake_invoke_hook(hook_type, payload, **_kwargs):
            if hook_type is HookType.GENERATION_BATCH_PRE_CALL:
                order.append("hook")
                modified = payload.model_copy(
                    update={
                        "model_options": {"temperature": 0.25},
                        "format": dict,
                        "tool_calls": True,
                    }
                )
                return (
                    PluginResult(continue_processing=True, modified_payload=modified),
                    modified,
                )
            return None, payload

        backend = RecordingBackend()

        with (
            patch("mellea.core.backend.has_plugins", return_value=True),
            patch("mellea.core.backend.invoke_hook", side_effect=fake_invoke_hook),
        ):
            await backend.generate_from_raw(
                [CBlock("a")],
                MagicMock(spec=Context),
                model_options={"temperature": 1.0},
                tool_calls=False,
            )

        assert order == ["hook", "generate"]
        assert captured_kwargs["model_options"] == {"temperature": 0.25}
        assert captured_kwargs["format"] is dict
        assert captured_kwargs["tool_calls"] is True

    async def test_batch_pre_call_mutation_propagates_with_real_plugin(self) -> None:
        """GENERATION_BATCH_PRE_CALL mutations survive the cpex policy layer."""
        captured_kwargs: dict[str, Any] = {}

        class RecordingBackend(_MockBackend):
            async def _generate_from_raw(self, actions, ctx, **kwargs):
                captured_kwargs.update(kwargs)
                return await super()._generate_from_raw(actions, ctx, **kwargs)

        @hook("generation_batch_pre_call")
        async def mutator(payload: Any, ctx: Any) -> Any:
            return PluginResult(
                continue_processing=True,
                modified_payload=payload.model_copy(
                    update={
                        "model_options": {"temperature": 0.25},
                        "format": dict,
                        "tool_calls": True,
                    }
                ),
            )

        register(mutator)
        backend = RecordingBackend()
        await backend.generate_from_raw(
            [CBlock("a")],
            MagicMock(spec=Context),
            model_options={"temperature": 1.0},
            tool_calls=False,
        )

        assert captured_kwargs["model_options"] == {"temperature": 0.25}
        assert captured_kwargs["format"] is dict
        assert captured_kwargs["tool_calls"] is True

    async def test_batch_post_call_fires_once_on_success(self) -> None:
        """GENERATION_BATCH_POST_CALL fires exactly once on success."""
        observed: list[Any] = []

        @hook("generation_batch_post_call")
        async def recorder(payload: Any, ctx: Any) -> Any:
            observed.append(payload)
            return None

        register(recorder)
        backend = _MockBackend()
        await backend.generate_from_raw([CBlock("a")], MagicMock(spec=Context))

        assert len(observed) == 1

    async def test_batch_post_call_payload_carries_results_and_usage(self) -> None:
        """Post-call payload carries model_outputs, usage, model, provider, latency_ms."""
        observed: list[Any] = []

        @hook("generation_batch_post_call")
        async def recorder(payload: Any, ctx: Any) -> Any:
            observed.append(payload)
            return None

        register(recorder)
        backend = _MockBackend()
        await backend.generate_from_raw(
            [CBlock("a"), CBlock("b")], MagicMock(spec=Context)
        )

        p = observed[0]
        # _MockBackend yields one MOT per action; usage is _MOCK_RAW_USAGE.
        assert len(p.model_outputs) == 2
        assert all(isinstance(m, ModelOutputThunk) for m in p.model_outputs)
        assert dict(p.usage) == _MOCK_RAW_USAGE
        assert p.model == "mock-model"
        assert p.provider == "mock-provider"
        assert p.latency_ms >= 0

    async def test_batch_post_call_generation_id_matches_pre_call(self) -> None:
        """Post-call's generation_id equals pre-call's generation_id for the same call."""
        observed_pre: list[Any] = []
        observed_post: list[Any] = []

        @hook("generation_batch_pre_call")
        async def pre_recorder(payload: Any, ctx: Any) -> Any:
            observed_pre.append(payload)
            return None

        @hook("generation_batch_post_call")
        async def post_recorder(payload: Any, ctx: Any) -> Any:
            observed_post.append(payload)
            return None

        register(pre_recorder)
        register(post_recorder)
        backend = _MockBackend()
        await backend.generate_from_raw([CBlock("a")], MagicMock(spec=Context))

        assert observed_pre[0].generation_id == observed_post[0].generation_id

    async def test_batch_error_fires_when_impl_raises(self) -> None:
        """GENERATION_BATCH_ERROR fires when _generate_from_raw raises; original re-raises."""
        boom = RuntimeError("boom")

        class _RaisingBackend(_MockBackend):
            async def _generate_from_raw(self, actions, ctx, **kwargs):
                raise boom

        observed_pre: list[Any] = []
        observed_err: list[Any] = []
        observed_post: list[Any] = []

        @hook("generation_batch_pre_call")
        async def pre_recorder(payload: Any, ctx: Any) -> Any:
            observed_pre.append(payload)
            return None

        @hook("generation_batch_error")
        async def err_recorder(payload: Any, ctx: Any) -> Any:
            observed_err.append(payload)
            return None

        @hook("generation_batch_post_call")
        async def post_recorder(payload: Any, ctx: Any) -> Any:
            observed_post.append(payload)
            return None

        register(pre_recorder)
        register(err_recorder)
        register(post_recorder)
        backend = _RaisingBackend()

        with pytest.raises(RuntimeError, match="boom"):
            await backend.generate_from_raw([CBlock("a")], MagicMock(spec=Context))

        # Error fires; post does not.
        assert len(observed_err) == 1
        assert len(observed_post) == 0
        err_payload = observed_err[0]
        assert err_payload.exception is boom
        assert err_payload.generation_id == observed_pre[0].generation_id
        assert err_payload.model == "mock-model"
        assert err_payload.provider == "mock-provider"
        assert err_payload.latency_ms >= 0

    async def test_batch_error_does_not_fire_on_success(self) -> None:
        """GENERATION_BATCH_ERROR does NOT fire on the success path."""
        observed_err: list[Any] = []

        @hook("generation_batch_error")
        async def err_recorder(payload: Any, ctx: Any) -> Any:
            observed_err.append(payload)
            return None

        register(err_recorder)
        backend = _MockBackend()
        await backend.generate_from_raw([CBlock("a")], MagicMock(spec=Context))

        assert len(observed_err) == 0


# ---------------------------------------------------------------------------
# Component hook call sites
# ---------------------------------------------------------------------------


class TestComponentHookCallSites:
    """Component hooks fire in ainstruct() and aact() in stdlib/functional.py."""

    async def test_component_pre_execute_fires_in_aact(self) -> None:
        """COMPONENT_PRE_EXECUTE fires in aact() before generation is called."""
        from mellea.stdlib.components import Instruction
        from mellea.stdlib.functional import aact

        observed: list[Any] = []

        @hook("component_pre_execute")
        async def recorder(payload: Any, ctx: Any) -> Any:
            observed.append(payload)
            return None

        register(recorder)
        backend = _MockBackend()
        ctx = SimpleContext()
        action = Instruction("Execute this")

        await aact(action, ctx, backend, strategy=None)
        assert len(observed) == 1

    async def test_component_pre_execute_payload_has_live_action(self) -> None:
        """COMPONENT_PRE_EXECUTE payload.action IS the same Component instance."""
        from mellea.stdlib.components import Instruction
        from mellea.stdlib.functional import aact

        observed: list[Any] = []

        @hook("component_pre_execute")
        async def recorder(payload: Any, ctx: Any) -> Any:
            observed.append(payload)
            return None

        register(recorder)
        backend = _MockBackend()
        ctx = SimpleContext()
        action = Instruction("Live reference test")

        await aact(action, ctx, backend, strategy=None)
        assert isinstance(observed[0].action, Instruction)
        assert observed[0].action._description.value == action._description.value  # type: ignore[union-attr]

    async def test_component_pre_execute_payload_component_type(self) -> None:
        """COMPONENT_PRE_EXECUTE payload.component_type matches the action class name."""
        from mellea.stdlib.components import Instruction
        from mellea.stdlib.functional import aact

        observed: list[Any] = []

        @hook("component_pre_execute")
        async def recorder(payload: Any, ctx: Any) -> Any:
            observed.append(payload)
            return None

        register(recorder)
        backend = _MockBackend()
        ctx = SimpleContext()
        action = Instruction("Type check")

        await aact(action, ctx, backend, strategy=None)
        assert observed[0].component_type == "Instruction"

    async def test_component_pre_execute_payload_has_context_view(self) -> None:
        """COMPONENT_PRE_EXECUTE payload.context_view mirrors view_for_generation()."""
        from mellea.stdlib.components import Instruction
        from mellea.stdlib.functional import aact

        observed: list[Any] = []

        @hook("component_pre_execute")
        async def recorder(payload: Any, ctx: Any) -> Any:
            observed.append(payload)
            return None

        register(recorder)
        backend = _MockBackend()
        ctx = SimpleContext.from_previous(SimpleContext(), CBlock("prior turn"))
        action = Instruction("Context view test")

        await aact(action, ctx, backend, strategy=None)
        assert observed[0].context_view is not None
        assert observed[0].context_view == ctx.view_for_generation()

    async def test_component_pre_execute_empty_context_gives_empty_list(self) -> None:
        """COMPONENT_PRE_EXECUTE on a fresh context gives an empty list, not None."""
        from mellea.stdlib.components import Instruction
        from mellea.stdlib.functional import aact

        observed: list[Any] = []

        @hook("component_pre_execute")
        async def recorder(payload: Any, ctx: Any) -> Any:
            observed.append(payload)
            return None

        register(recorder)
        backend = _MockBackend()
        ctx = SimpleContext()
        action = Instruction("Fresh context")

        await aact(action, ctx, backend, strategy=None)
        assert observed[0].context_view is not None
        assert observed[0].context_view == []

    async def test_component_post_success_fires_in_aact(self) -> None:
        """COMPONENT_POST_SUCCESS fires in aact() after successful generation."""
        from mellea.stdlib.components import Instruction
        from mellea.stdlib.functional import aact

        observed: list[Any] = []

        @hook("component_post_success")
        async def recorder(payload: Any, ctx: Any) -> Any:
            observed.append(payload)
            return None

        register(recorder)
        backend = _MockBackend()
        ctx = SimpleContext()
        action = Instruction("Success test")

        _result, _new_ctx = await aact(action, ctx, backend, strategy=None)
        assert len(observed) == 1

    async def test_component_post_success_payload_has_correct_result_and_contexts(
        self,
    ) -> None:
        """COMPONENT_POST_SUCCESS payload carries result, context_before, context_after."""
        from mellea.stdlib.components import Instruction
        from mellea.stdlib.functional import aact

        observed: list[Any] = []

        @hook("component_post_success")
        async def recorder(payload: Any, ctx: Any) -> Any:
            observed.append(payload)
            return None

        register(recorder)
        backend = _MockBackend()
        ctx = SimpleContext()
        action = Instruction("Payload check")

        _result, _new_ctx = await aact(action, ctx, backend, strategy=None)

        p = observed[0]

        assert p.result is not None
        assert p.context_before is not None
        assert p.context_after is not None
        assert p.action is not None
        assert p.latency_ms >= 0


# ---------------------------------------------------------------------------
# Sampling hook call sites
# ---------------------------------------------------------------------------


class TestSamplingHookCallSites:
    """SAMPLING_LOOP_START, SAMPLING_ITERATION, SAMPLING_LOOP_END fire in
    BaseSamplingStrategy.sample()."""

    async def test_sampling_loop_start_fires(self) -> None:
        """SAMPLING_LOOP_START fires when RejectionSamplingStrategy.sample() begins."""
        from mellea.stdlib.components import Instruction
        from mellea.stdlib.sampling.base import RejectionSamplingStrategy

        observed: list[Any] = []

        @hook("sampling_loop_start")
        async def recorder(payload: Any, ctx: Any) -> Any:
            observed.append(payload)
            return None

        register(recorder)
        backend = _MockBackend()
        ctx = SimpleContext()
        strategy = RejectionSamplingStrategy(loop_budget=1)

        await strategy.sample(
            Instruction("Sample test"),
            context=ctx,
            backend=backend,
            requirements=[],
            format=None,
            model_options=None,
            tool_calls=False,
            show_progress=False,
        )
        assert len(observed) == 1

    async def test_sampling_loop_start_payload_has_strategy_name(self) -> None:
        """SAMPLING_LOOP_START payload.strategy_name contains the strategy class name."""
        from mellea.stdlib.components import Instruction
        from mellea.stdlib.sampling.base import RejectionSamplingStrategy

        observed: list[Any] = []

        @hook("sampling_loop_start")
        async def recorder(payload: Any, ctx: Any) -> Any:
            observed.append(payload)
            return None

        register(recorder)
        backend = _MockBackend()
        ctx = SimpleContext()
        strategy = RejectionSamplingStrategy(loop_budget=1)

        await strategy.sample(
            Instruction("Strategy name test"),
            context=ctx,
            backend=backend,
            requirements=[],
            format=None,
            model_options=None,
            tool_calls=False,
            show_progress=False,
        )
        assert "RejectionSampling" in observed[0].strategy_name

    async def test_sampling_loop_start_payload_has_correct_loop_budget(self) -> None:
        """SAMPLING_LOOP_START payload.loop_budget matches the strategy's loop_budget."""
        from mellea.stdlib.components import Instruction
        from mellea.stdlib.sampling.base import RejectionSamplingStrategy

        observed: list[Any] = []

        @hook("sampling_loop_start")
        async def recorder(payload: Any, ctx: Any) -> Any:
            observed.append(payload)
            return None

        register(recorder)
        backend = _MockBackend()
        ctx = SimpleContext()
        strategy = RejectionSamplingStrategy(loop_budget=3)

        await strategy.sample(
            Instruction("Budget test"),
            context=ctx,
            backend=backend,
            requirements=[],
            format=None,
            model_options=None,
            tool_calls=False,
            show_progress=False,
        )
        assert observed[0].loop_budget == 3

    @pytest.mark.parametrize("bad_budget", [0, -1])
    async def test_sampling_loop_start_rejects_non_positive_loop_budget(
        self, bad_budget: int
    ) -> None:
        """A SAMPLING_LOOP_START hook that returns loop_budget < 1 must raise ValueError.

        The constructor enforces loop_budget >= 1 but a hook can override it
        post-construction; without explicit validation this collapses
        total_possible_generations to 0, no slices are produced, and the user
        sees an opaque AssertionError from SamplingResult.__init__ instead of
        a clear cause.
        """
        from mellea.stdlib.components import Instruction
        from mellea.stdlib.sampling.base import RejectionSamplingStrategy

        @hook("sampling_loop_start")
        async def shrink_budget(payload: Any, ctx: Any) -> Any:
            return PluginResult(
                continue_processing=True,
                modified_payload=payload.model_copy(update={"loop_budget": bad_budget}),
            )

        register(shrink_budget)
        backend = _MockBackend()
        ctx = SimpleContext()
        strategy = RejectionSamplingStrategy(loop_budget=1)

        with pytest.raises(ValueError, match="non-positive loop_budget"):
            await strategy.sample(
                Instruction("Bad budget test"),
                context=ctx,
                backend=backend,
                requirements=[],
                format=None,
                model_options=None,
                tool_calls=False,
                show_progress=False,
            )

    async def test_sampling_iteration_fires_once_per_loop_iteration(self) -> None:
        """SAMPLING_ITERATION fires once per loop iteration."""
        from mellea.stdlib.components import Instruction
        from mellea.stdlib.sampling.base import RejectionSamplingStrategy

        observed: list[Any] = []

        @hook("sampling_iteration")
        async def recorder(payload: Any, ctx: Any) -> Any:
            observed.append(payload)
            return None

        register(recorder)
        backend = _MockBackend()
        ctx = SimpleContext()
        # With loop_budget=1 and no requirements, exactly 1 iteration runs
        strategy = RejectionSamplingStrategy(loop_budget=1)

        await strategy.sample(
            Instruction("Iteration test"),
            context=ctx,
            backend=backend,
            requirements=[],
            format=None,
            model_options=None,
            tool_calls=False,
            show_progress=False,
        )
        assert len(observed) == 1
        assert observed[0].iteration == 1
        assert observed[0].all_validations_passed is True  # no requirements → all pass

    async def test_sampling_iteration_ids_unique_under_concurrency(self) -> None:
        """Each generate/validate of the base sampling strategy should report a unique `iteration` to the sampling iteration hook.

        With `concurrency_budget=3, loop_budget=2`, six generations run; their `SAMPLING_ITERATION`
        payloads must carry six distinct `iteration` values (computed as `subsample_index * loop_budget + i + 1`),
        not three duplicated `{1, 2}` pairs.
        """

        observed: list[int] = []

        @hook("sampling_iteration")
        async def recorder(payload: Any, ctx: Any) -> Any:
            observed.append(payload.iteration)
            return None

        register(recorder)
        backend = _MockBackend()
        ctx = SimpleContext()

        always_fail = Requirement(
            description="always fails",
            validation_fn=lambda _ctx: ValidationResult(
                result=False, reason="forced failure"
            ),
        )
        loop_budget = 2
        concurrency_budget = 3
        strategy = RejectionSamplingStrategy(
            loop_budget=loop_budget, concurrency_budget=concurrency_budget
        )

        await strategy.sample(
            Instruction("Concurrency iteration id test"),
            context=ctx,
            backend=backend,
            requirements=[always_fail],
            format=None,
            model_options=None,
            tool_calls=False,
            show_progress=False,
        )

        expected = set(range(1, loop_budget * concurrency_budget + 1))
        assert set(observed) == expected, (
            f"iteration ids should be {sorted(expected)} (one per generation, unique "
            f"across subsamples), got {sorted(observed)}"
        )

    async def test_sampling_loop_end_fires_on_success_path(self) -> None:
        """SAMPLING_LOOP_END fires with success=True when sampling succeeds."""
        from mellea.stdlib.components import Instruction
        from mellea.stdlib.sampling.base import RejectionSamplingStrategy

        observed: list[Any] = []

        @hook("sampling_loop_end")
        async def recorder(payload: Any, ctx: Any) -> Any:
            observed.append(payload)
            return None

        register(recorder)
        backend = _MockBackend()
        ctx = SimpleContext()
        strategy = RejectionSamplingStrategy(loop_budget=1)

        await strategy.sample(
            Instruction("End test"),
            context=ctx,
            backend=backend,
            requirements=[],
            format=None,
            model_options=None,
            tool_calls=False,
            show_progress=False,
        )
        assert len(observed) == 1
        assert observed[0].success is True

    async def test_sampling_loop_end_success_payload_has_final_result_and_context(
        self,
    ) -> None:
        """SAMPLING_LOOP_END success payload has final_result and final_context populated."""
        from mellea.stdlib.components import Instruction
        from mellea.stdlib.sampling.base import RejectionSamplingStrategy

        observed: list[Any] = []

        @hook("sampling_loop_end")
        async def recorder(payload: Any, ctx: Any) -> Any:
            observed.append(payload)
            return None

        register(recorder)
        backend = _MockBackend()
        ctx = SimpleContext()
        strategy = RejectionSamplingStrategy(loop_budget=1)

        await strategy.sample(
            Instruction("Final payload test"),
            context=ctx,
            backend=backend,
            requirements=[],
            format=None,
            model_options=None,
            tool_calls=False,
            show_progress=False,
        )
        p = observed[0]
        assert p.final_result is not None
        assert p.final_context is not None
        assert isinstance(p.all_results, list)
        assert len(p.all_results) == 1  # one iteration, one result

    async def test_sampling_loop_end_context_available_on_payload(self) -> None:
        """On success, SAMPLING_LOOP_END payload carries final_context (post-generation)."""
        from mellea.stdlib.components import Instruction
        from mellea.stdlib.sampling.base import RejectionSamplingStrategy

        observed_ctxs: list[Any] = []

        @hook("sampling_loop_end")
        async def recorder(payload: Any, ctx: Any) -> Any:
            observed_ctxs.append(payload.final_context)
            return None

        register(recorder)
        backend = _MockBackend()
        original_ctx = SimpleContext()
        strategy = RejectionSamplingStrategy(loop_budget=1)

        await strategy.sample(
            Instruction("Context test"),
            context=original_ctx,
            backend=backend,
            requirements=[],
            format=None,
            model_options=None,
            tool_calls=False,
            show_progress=False,
        )
        # On success, the payload's final_context should be result_ctx (not original_ctx)
        if observed_ctxs:
            assert observed_ctxs[0] is not original_ctx, (
                "Success path: payload.final_context should be result_ctx, not the original input context"
            )

    async def test_all_three_sampling_hooks_fire_in_order(self) -> None:
        """SAMPLING_LOOP_START → SAMPLING_ITERATION → SAMPLING_LOOP_END order."""
        from mellea.stdlib.components import Instruction
        from mellea.stdlib.sampling.base import RejectionSamplingStrategy

        order: list[str] = []

        @hook("sampling_loop_start")
        async def h1(payload: Any, ctx: Any) -> Any:
            order.append("loop_start")
            return None

        @hook("sampling_iteration")
        async def h2(payload: Any, ctx: Any) -> Any:
            order.append("iteration")
            return None

        @hook("sampling_loop_end")
        async def h3(payload: Any, ctx: Any) -> Any:
            order.append("loop_end")
            return None

        register(h1)
        register(h2)
        register(h3)
        backend = _MockBackend()
        ctx = SimpleContext()
        strategy = RejectionSamplingStrategy(loop_budget=1)

        await strategy.sample(
            Instruction("Order test"),
            context=ctx,
            backend=backend,
            requirements=[],
            format=None,
            model_options=None,
            tool_calls=False,
            show_progress=False,
        )
        assert order == ["loop_start", "iteration", "loop_end"]

    async def test_sampling_repair_skipped_on_final_iteration(self) -> None:
        """SAMPLING_REPAIR fires only between iterations, never after the last one.

        With loop_budget=N and all-failing requirements, repair runs N-1 times:
        once after each failed attempt that has a successor, but not after the
        final attempt (its repair output would be discarded).
        """
        observed: list[Any] = []

        @hook("sampling_repair")
        async def recorder(payload: Any, ctx: Any) -> Any:
            observed.append(payload)
            return None

        register(recorder)
        backend = _MockBackend()
        ctx = SimpleContext()

        always_fail = Requirement(
            description="always fails",
            validation_fn=lambda _ctx: ValidationResult(
                result=False, reason="forced failure"
            ),
        )
        loop_budget = 3
        strategy = RejectionSamplingStrategy(loop_budget=loop_budget)

        await strategy.sample(
            Instruction("Repair-skip test"),
            context=ctx,
            backend=backend,
            requirements=[always_fail],
            format=None,
            model_options=None,
            tool_calls=False,
            show_progress=False,
        )

        assert len(observed) == loop_budget - 1, (
            f"repair should fire N-1 times for N iterations, got {len(observed)}"
        )

    async def test_sampling_repair_not_fired_when_loop_budget_is_one(self) -> None:
        """With loop_budget=1, SAMPLING_REPAIR must never fire even on validation failure."""
        observed: list[Any] = []

        @hook("sampling_repair")
        async def recorder(payload: Any, ctx: Any) -> Any:
            observed.append(payload)
            return None

        register(recorder)
        backend = _MockBackend()
        ctx = SimpleContext()

        always_fail = Requirement(
            description="always fails",
            validation_fn=lambda _ctx: ValidationResult(result=False),
        )
        strategy = RejectionSamplingStrategy(loop_budget=1)

        await strategy.sample(
            Instruction("Single-iteration test"),
            context=ctx,
            backend=backend,
            requirements=[always_fail],
            format=None,
            model_options=None,
            tool_calls=False,
            show_progress=False,
        )

        assert observed == [], (
            "loop_budget=1 should never invoke repair (no next iteration to feed)"
        )

    async def test_sampling_id_correlates_iteration_and_repair(self) -> None:
        """All sampling hooks in one loop share the same `sampling_id`."""
        observed: list[tuple[str, Any]] = []

        @hook("sampling_loop_start")
        async def record_start(payload: Any, ctx: Any) -> Any:
            observed.append(("start", payload))
            return None

        @hook("sampling_iteration")
        async def record_iteration(payload: Any, ctx: Any) -> Any:
            observed.append(("iteration", payload))
            return None

        @hook("sampling_repair")
        async def record_repair(payload: Any, ctx: Any) -> Any:
            observed.append(("repair", payload))
            return None

        @hook("sampling_loop_end")
        async def record_end(payload: Any, ctx: Any) -> Any:
            observed.append(("end", payload))
            return None

        register(record_start)
        register(record_iteration)
        register(record_repair)
        register(record_end)

        backend = _MockBackend()
        strategy = RejectionSamplingStrategy(loop_budget=2)
        always_fail = Requirement(
            description="always fails",
            validation_fn=lambda _ctx: ValidationResult(result=False),
        )

        await strategy.sample(
            Instruction("sampling-id correlation"),
            context=SimpleContext(),
            backend=backend,
            requirements=[always_fail],
            show_progress=False,
        )

        assert [kind for kind, _ in observed] == [
            "start",
            "iteration",
            "repair",
            "iteration",
            "end",
        ]
        sampling_ids = {payload.sampling_id for _, payload in observed}
        assert len(sampling_ids) == 1


class TestBudgetForcingHookCallSites:
    """Budget forcing emits sampling iteration/repair hooks at its loop call sites."""

    @staticmethod
    def _always_fail_requirement() -> Requirement:
        """Return a requirement that always fails."""
        return Requirement(
            description="always fails",
            validation_fn=lambda _ctx: ValidationResult(
                result=False, reason="forced failure"
            ),
        )

    @staticmethod
    def _always_pass_requirement() -> Requirement:
        """Return a requirement that always passes."""
        return Requirement(
            description="always passes",
            validation_fn=lambda _ctx: ValidationResult(result=True),
        )

    @staticmethod
    def _make_budget_forcing_result(value: str) -> ModelOutputThunk:
        """Build a budget-forcing result thunk with the required generate log."""
        mot = ModelOutputThunk(value=value)
        glog = GenerateLog()
        glog.prompt = "mocked formatted prompt"
        glog.is_final_result = False
        mot._generate_log = glog
        mot._gen.start = datetime.datetime.now()
        return mot

    async def test_iteration_fires_once_per_budget_forcing_loop(self) -> None:
        """Budget forcing emits one iteration hook per loop attempt."""
        observed: list[Any] = []

        @hook("sampling_iteration")
        async def recorder(payload: Any, ctx: Any) -> Any:
            observed.append(payload)
            return None

        register(recorder)
        backend = _MockBackend()
        strategy = BudgetForcingSamplingStrategy(loop_budget=2, requirements=None)

        async def fake_think_budget_forcing(
            *args: Any, **kwargs: Any
        ) -> ModelOutputThunk:
            return self._make_budget_forcing_result("budget output")

        with (
            patch("mellea.backends.ollama.OllamaModelBackend", _MockBackend),
            patch(
                "mellea.stdlib.sampling.budget_forcing.think_budget_forcing",
                side_effect=fake_think_budget_forcing,
            ),
        ):
            await strategy.sample(
                Instruction("Budget forcing iteration test"),
                context=SimpleContext(),
                backend=backend,
                requirements=[self._always_fail_requirement()],
                show_progress=False,
            )

        assert [payload.iteration for payload in observed] == [1, 2]

    async def test_iteration_fires_on_success_path(self) -> None:
        """Budget forcing still emits iteration on the passing attempt."""
        iter_observed: list[Any] = []
        repair_observed: list[Any] = []

        @hook("sampling_iteration")
        async def record_iteration(payload: Any, ctx: Any) -> Any:
            iter_observed.append(payload)
            return None

        @hook("sampling_repair")
        async def record_repair(payload: Any, ctx: Any) -> Any:
            repair_observed.append(payload)
            return None

        register(record_iteration)
        register(record_repair)
        backend = _MockBackend()
        strategy = BudgetForcingSamplingStrategy(loop_budget=2, requirements=None)

        async def fake_think_budget_forcing(
            *args: Any, **kwargs: Any
        ) -> ModelOutputThunk:
            return self._make_budget_forcing_result("budget output")

        with (
            patch("mellea.backends.ollama.OllamaModelBackend", _MockBackend),
            patch(
                "mellea.stdlib.sampling.budget_forcing.think_budget_forcing",
                side_effect=fake_think_budget_forcing,
            ),
        ):
            await strategy.sample(
                Instruction("Budget forcing success test"),
                context=SimpleContext(),
                backend=backend,
                requirements=[self._always_pass_requirement()],
                show_progress=False,
            )

        assert len(iter_observed) == 1
        assert iter_observed[0].iteration == 1
        assert iter_observed[0].all_validations_passed is True
        assert repair_observed == []

    async def test_repair_fires_between_iterations_not_after_last(self) -> None:
        """Budget forcing emits repair only when another iteration will follow."""
        observed: list[Any] = []

        @hook("sampling_repair")
        async def recorder(payload: Any, ctx: Any) -> Any:
            observed.append(payload)
            return None

        register(recorder)
        backend = _MockBackend()
        strategy = BudgetForcingSamplingStrategy(loop_budget=2, requirements=None)

        async def fake_think_budget_forcing(
            *args: Any, **kwargs: Any
        ) -> ModelOutputThunk:
            return self._make_budget_forcing_result("budget output")

        with (
            patch("mellea.backends.ollama.OllamaModelBackend", _MockBackend),
            patch(
                "mellea.stdlib.sampling.budget_forcing.think_budget_forcing",
                side_effect=fake_think_budget_forcing,
            ),
        ):
            await strategy.sample(
                Instruction("Budget forcing repair count test"),
                context=SimpleContext(),
                backend=backend,
                requirements=[self._always_fail_requirement()],
                show_progress=False,
            )

        assert len(observed) == 1
        assert observed[0].repair_iteration == 1

    async def test_repair_type_is_budget_forcing(self) -> None:
        """Budget forcing repair payload advertises its repair type."""
        observed: list[Any] = []

        @hook("sampling_repair")
        async def recorder(payload: Any, ctx: Any) -> Any:
            observed.append(payload)
            return None

        register(recorder)
        backend = _MockBackend()
        strategy = BudgetForcingSamplingStrategy(loop_budget=2, requirements=None)

        async def fake_think_budget_forcing(
            *args: Any, **kwargs: Any
        ) -> ModelOutputThunk:
            return self._make_budget_forcing_result("budget output")

        with (
            patch("mellea.backends.ollama.OllamaModelBackend", _MockBackend),
            patch(
                "mellea.stdlib.sampling.budget_forcing.think_budget_forcing",
                side_effect=fake_think_budget_forcing,
            ),
        ):
            await strategy.sample(
                Instruction("Budget forcing repair type test"),
                context=SimpleContext(),
                backend=backend,
                requirements=[self._always_fail_requirement()],
                show_progress=False,
            )

        assert len(observed) == 1
        assert observed[0].repair_type == "budget_forcing"

    async def test_sampling_id_correlates_all_events(self) -> None:
        """Budget forcing loop hooks all share a single `sampling_id`."""
        observed: list[tuple[str, Any]] = []

        @hook("sampling_loop_start")
        async def record_start(payload: Any, ctx: Any) -> Any:
            observed.append(("start", payload))
            return None

        @hook("sampling_iteration")
        async def record_iteration(payload: Any, ctx: Any) -> Any:
            observed.append(("iteration", payload))
            return None

        @hook("sampling_repair")
        async def record_repair(payload: Any, ctx: Any) -> Any:
            observed.append(("repair", payload))
            return None

        @hook("sampling_loop_end")
        async def record_end(payload: Any, ctx: Any) -> Any:
            observed.append(("end", payload))
            return None

        register(record_start)
        register(record_iteration)
        register(record_repair)
        register(record_end)
        backend = _MockBackend()
        strategy = BudgetForcingSamplingStrategy(loop_budget=2, requirements=None)

        async def fake_think_budget_forcing(
            *args: Any, **kwargs: Any
        ) -> ModelOutputThunk:
            return self._make_budget_forcing_result("budget output")

        with (
            patch("mellea.backends.ollama.OllamaModelBackend", _MockBackend),
            patch(
                "mellea.stdlib.sampling.budget_forcing.think_budget_forcing",
                side_effect=fake_think_budget_forcing,
            ),
        ):
            await strategy.sample(
                Instruction("Budget forcing sampling-id test"),
                context=SimpleContext(),
                backend=backend,
                requirements=[self._always_fail_requirement()],
                show_progress=False,
            )

        assert [kind for kind, _ in observed] == [
            "start",
            "iteration",
            "repair",
            "iteration",
            "end",
        ]
        sampling_ids = {payload.sampling_id for _, payload in observed}
        assert len(sampling_ids) == 1


class TestSOFAIHookCallSites:
    """SOFAI emits iteration/repair hooks across S1 and S2 phases."""

    @staticmethod
    def _always_fail_requirement() -> Requirement:
        """Return a requirement that always fails."""
        return Requirement(
            description="always fails",
            validation_fn=lambda _ctx: ValidationResult(
                result=False, reason="forced failure"
            ),
        )

    @staticmethod
    def _base_context() -> ChatContext:
        """Return a minimal valid chat context for SOFAI."""
        return ChatContext().add(Message(role="user", content="Solve this"))

    async def test_s1_iteration_fires_each_attempt(self) -> None:
        """SOFAI emits iteration hooks for each S1 attempt before S2 escalation."""
        observed: list[Any] = []

        @hook("sampling_iteration")
        async def recorder(payload: Any, ctx: Any) -> Any:
            observed.append(payload)
            return None

        register(recorder)
        strategy = SOFAISamplingStrategy(
            s1_solver_backend=_MockBackend(),
            s2_solver_backend=_MockBackend(),
            loop_budget=2,
            s2_solver_mode="fresh_start",
        )

        await strategy.sample(
            Message(role="user", content="Question"),
            context=self._base_context(),
            backend=_MockBackend(),
            requirements=[self._always_fail_requirement()],
            show_progress=False,
        )

        assert [payload.iteration for payload in observed[:2]] == [1, 2]

    async def test_s2_iteration_fires_as_final_iteration(self) -> None:
        """SOFAI emits the S2 escalation as the final iteration number."""
        observed: list[Any] = []

        @hook("sampling_iteration")
        async def recorder(payload: Any, ctx: Any) -> Any:
            observed.append(payload)
            return None

        register(recorder)
        strategy = SOFAISamplingStrategy(
            s1_solver_backend=_MockBackend(),
            s2_solver_backend=_MockBackend(),
            loop_budget=2,
            s2_solver_mode="fresh_start",
        )

        await strategy.sample(
            Message(role="user", content="Question"),
            context=self._base_context(),
            backend=_MockBackend(),
            requirements=[self._always_fail_requirement()],
            show_progress=False,
        )

        assert len(observed) == 3

    async def test_s1_repair_fires_not_after_last_s1_attempt(self) -> None:
        """SOFAI emits one S1 repair between two failed S1 attempts."""
        observed: list[Any] = []

        @hook("sampling_repair")
        async def recorder(payload: Any, ctx: Any) -> Any:
            observed.append(payload)
            return None

        register(recorder)
        strategy = SOFAISamplingStrategy(
            s1_solver_backend=_MockBackend(),
            s2_solver_backend=_MockBackend(),
            loop_budget=2,
            s2_solver_mode="fresh_start",
        )

        await strategy.sample(
            Message(role="user", content="Question"),
            context=self._base_context(),
            backend=_MockBackend(),
            requirements=[self._always_fail_requirement()],
            show_progress=False,
        )

        assert len(observed) == 1
        assert observed[0].repair_iteration == 1
        assert observed[0].repair_type == "sofai_s1_feedback"

    async def test_no_s2_repair_event(self) -> None:
        """Immediate S2 escalation does not emit a repair event."""
        observed: list[Any] = []

        @hook("sampling_repair")
        async def recorder(payload: Any, ctx: Any) -> Any:
            observed.append(payload)
            return None

        register(recorder)
        strategy = SOFAISamplingStrategy(
            s1_solver_backend=_MockBackend(),
            s2_solver_backend=_MockBackend(),
            loop_budget=1,
            s2_solver_mode="fresh_start",
        )

        await strategy.sample(
            Message(role="user", content="Question"),
            context=self._base_context(),
            backend=_MockBackend(),
            requirements=[self._always_fail_requirement()],
            show_progress=False,
        )

        assert observed == []

    async def test_effective_budget_respected_repair_guard(self) -> None:
        """SOFAI skips repair when a loop-start hook shrinks the effective budget to 1."""
        observed: list[Any] = []

        @hook("sampling_loop_start")
        async def shrink_budget(payload: Any, ctx: Any) -> Any:
            return PluginResult(
                continue_processing=True,
                modified_payload=payload.model_copy(update={"loop_budget": 1}),
            )

        @hook("sampling_repair")
        async def record_repair(payload: Any, ctx: Any) -> Any:
            observed.append(payload)
            return None

        register(shrink_budget)
        register(record_repair)
        strategy = SOFAISamplingStrategy(
            s1_solver_backend=_MockBackend(),
            s2_solver_backend=_MockBackend(),
            loop_budget=5,
            s2_solver_mode="fresh_start",
        )

        await strategy.sample(
            Message(role="user", content="Question"),
            context=self._base_context(),
            backend=_MockBackend(),
            requirements=[self._always_fail_requirement()],
            show_progress=False,
        )

        assert observed == []

    async def test_sampling_id_correlates_all_sofai_events(self) -> None:
        """SOFAI loop hooks all share one `sampling_id`."""
        observed: list[tuple[str, Any]] = []

        @hook("sampling_loop_start")
        async def record_start(payload: Any, ctx: Any) -> Any:
            observed.append(("start", payload))
            return None

        @hook("sampling_iteration")
        async def record_iteration(payload: Any, ctx: Any) -> Any:
            observed.append(("iteration", payload))
            return None

        @hook("sampling_loop_end")
        async def record_end(payload: Any, ctx: Any) -> Any:
            observed.append(("end", payload))
            return None

        register(record_start)
        register(record_iteration)
        register(record_end)
        strategy = SOFAISamplingStrategy(
            s1_solver_backend=_MockBackend(),
            s2_solver_backend=_MockBackend(),
            loop_budget=1,
            s2_solver_mode="fresh_start",
        )

        await strategy.sample(
            Message(role="user", content="Question"),
            context=self._base_context(),
            backend=_MockBackend(),
            requirements=[self._always_fail_requirement()],
            show_progress=False,
        )

        assert [kind for kind, _ in observed] == [
            "start",
            "iteration",
            "iteration",
            "end",
        ]
        sampling_ids = {payload.sampling_id for _, payload in observed}
        assert len(sampling_ids) == 1


# ---------------------------------------------------------------------------
# Majority-voting lifecycle hooks — regression for sampling_id propagation
# ---------------------------------------------------------------------------


class TestMajorityVotingHookCallSites:
    """Majority-voting emits exactly one start/end pair per sample() call,
    and all hooks share the same sampling_id.

    This is the key regression for the change from super().sample() to
    super()._sample_impl() in BaseMBRDSampling: the inner fan-out must reuse
    the outer sampling_id rather than minting N independent loops.
    """

    async def test_majority_voting_emits_single_loop_start_and_end(self) -> None:
        """MBRDRougeLStrategy.sample() fires exactly one loop_start and one loop_end."""
        from mellea.stdlib.sampling.majority_voting import MBRDRougeLStrategy

        start_events: list[Any] = []
        end_events: list[Any] = []

        @hook("sampling_loop_start")
        async def record_start(payload: Any, ctx: Any) -> Any:
            start_events.append(payload)
            return None

        @hook("sampling_loop_end")
        async def record_end(payload: Any, ctx: Any) -> Any:
            end_events.append(payload)
            return None

        register(record_start)
        register(record_end)

        backend = _MockBackend()
        strategy = MBRDRougeLStrategy(number_of_samples=3, loop_budget=1)

        await strategy.sample(
            Instruction("majority voting hook test"),
            context=SimpleContext(),
            backend=backend,
            requirements=[],
            show_progress=False,
        )

        assert len(start_events) == 1, (
            f"Expected exactly one sampling_loop_start, got {len(start_events)}"
        )
        assert len(end_events) == 1, (
            f"Expected exactly one sampling_loop_end, got {len(end_events)}"
        )

    async def test_majority_voting_all_hooks_share_sampling_id(self) -> None:
        """All hooks emitted during MBRDRougeLStrategy.sample() share one sampling_id.

        This is the core invariant broken when the inner fan-out called
        super().sample() (which minted a fresh UUID per inner call) and fixed by
        calling super()._sample_impl() with the outer sampling_id instead.
        """
        from mellea.stdlib.sampling.majority_voting import MBRDRougeLStrategy

        observed: list[tuple[str, Any]] = []

        @hook("sampling_loop_start")
        async def record_start(payload: Any, ctx: Any) -> Any:
            observed.append(("start", payload))
            return None

        @hook("sampling_iteration")
        async def record_iteration(payload: Any, ctx: Any) -> Any:
            observed.append(("iteration", payload))
            return None

        @hook("sampling_loop_end")
        async def record_end(payload: Any, ctx: Any) -> Any:
            observed.append(("end", payload))
            return None

        register(record_start)
        register(record_iteration)
        register(record_end)

        backend = _MockBackend()
        number_of_samples = 3
        strategy = MBRDRougeLStrategy(
            number_of_samples=number_of_samples, loop_budget=1
        )

        await strategy.sample(
            Instruction("majority voting sampling_id test"),
            context=SimpleContext(),
            backend=backend,
            requirements=[],
            show_progress=False,
        )

        kinds = [kind for kind, _ in observed]
        # One start, N iterations (one per fan-out sample), one end.
        assert kinds[0] == "start"
        assert kinds[-1] == "end"
        assert kinds.count("start") == 1
        assert kinds.count("end") == 1
        assert kinds.count("iteration") == number_of_samples

        sampling_ids = {payload.sampling_id for _, payload in observed}
        assert len(sampling_ids) == 1, (
            f"All events must share one sampling_id; got {sampling_ids}"
        )

    async def test_majority_voting_repair_path_hooks_share_sampling_id(self) -> None:
        """Iteration and repair hooks from inner fan-out loops share the outer sampling_id.

        With number_of_samples=N and loop_budget=B and an always-failing requirement:
          - each of the N inner rejection loops runs B iterations and (B-1) repairs
          - total: N*B iteration events, N*(B-1) repair events
          - exactly one start/end pair wrapping the whole operation
          - every hook payload carries the same sampling_id
        """
        from mellea.stdlib.sampling.majority_voting import MBRDRougeLStrategy

        observed: list[tuple[str, Any]] = []

        @hook("sampling_loop_start")
        async def record_start(payload: Any, ctx: Any) -> Any:
            observed.append(("start", payload))
            return None

        @hook("sampling_iteration")
        async def record_iteration(payload: Any, ctx: Any) -> Any:
            observed.append(("iteration", payload))
            return None

        @hook("sampling_repair")
        async def record_repair(payload: Any, ctx: Any) -> Any:
            observed.append(("repair", payload))
            return None

        @hook("sampling_loop_end")
        async def record_end(payload: Any, ctx: Any) -> Any:
            observed.append(("end", payload))
            return None

        register(record_start)
        register(record_iteration)
        register(record_repair)
        register(record_end)

        number_of_samples = 2
        loop_budget = 2
        always_fail = Requirement(
            description="always fails",
            validation_fn=lambda _ctx: ValidationResult(
                result=False, reason="forced failure"
            ),
        )
        strategy = MBRDRougeLStrategy(
            number_of_samples=number_of_samples, loop_budget=loop_budget
        )

        await strategy.sample(
            Instruction("majority voting repair-path hook test"),
            context=SimpleContext(),
            backend=_MockBackend(),
            requirements=[always_fail],
            show_progress=False,
        )

        kinds = [kind for kind, _ in observed]

        # Exactly one start/end pair.
        assert kinds[0] == "start"
        assert kinds[-1] == "end"
        assert kinds.count("start") == 1
        assert kinds.count("end") == 1

        # N*B iteration events: each inner rejection loop fires loop_budget iterations.
        expected_iterations = number_of_samples * loop_budget
        assert kinds.count("iteration") == expected_iterations, (
            f"Expected {expected_iterations} iteration events "
            f"({number_of_samples} samples × {loop_budget} iterations), "
            f"got {kinds.count('iteration')}"
        )

        # N*(B-1) repair events: repair fires between iterations but not after the last.
        expected_repairs = number_of_samples * (loop_budget - 1)
        assert kinds.count("repair") == expected_repairs, (
            f"Expected {expected_repairs} repair events "
            f"({number_of_samples} samples × {loop_budget - 1} repairs), "
            f"got {kinds.count('repair')}"
        )

        # Every event carries the same sampling_id.
        sampling_ids = {payload.sampling_id for _, payload in observed}
        assert len(sampling_ids) == 1, (
            f"All events must share one sampling_id; got {sampling_ids}"
        )


# ---------------------------------------------------------------------------
# Sampling wrapper contract (mock strategy overriding only _sample_impl)
# ---------------------------------------------------------------------------


def _make_minimal_sampling_result(
    context: Context, *, success: bool = True
) -> SamplingResult:
    """Build the smallest valid SamplingResult from a pre-computed thunk.

    Used by TestSamplingWrapperContract to return a handcrafted result without
    running any real sampling machinery.
    """
    mot = ModelOutputThunk(value="stub output")
    glog = GenerateLog()
    glog.prompt = "stub prompt"
    mot._generate_log = glog
    mot._gen.start = datetime.datetime.now()
    computed = ComputedModelOutputThunk(mot)
    return SamplingResult(
        result_index=0,
        success=success,
        sample_generations=[computed],
        sample_validations=[[]],
        sample_actions=[CBlock("stub action")],
        sample_contexts=[context],
    )


class _MinimalStrategy(RejectionSamplingStrategy):
    """A strategy that delegates all work to a user-supplied coroutine.

    Overrides only `_sample_impl` so the `SamplingStrategy.sample` wrapper is
    the sole source of sampling lifecycle hooks.
    """

    def __init__(self, impl_fn, loop_budget: int = 1) -> None:
        """Initialize _MinimalStrategy with a custom _sample_impl coroutine and loop budget."""
        super().__init__(loop_budget=loop_budget)
        self._impl_fn = impl_fn

    async def _sample_impl(self, action, context, backend, requirements, **kwargs):  # type: ignore[override]
        """Delegate to user-supplied impl_fn."""
        return await self._impl_fn(action, context, backend, requirements, **kwargs)


class TestSamplingWrapperContract:
    """The `SamplingStrategy.sample` wrapper fires lifecycle hooks regardless of subclass implementation."""

    async def test_loop_start_fires_from_wrapper_not_subclass(self) -> None:
        """sampling_loop_start fires even when only _sample_impl is overridden."""
        observed: list[Any] = []

        @hook("sampling_loop_start")
        async def recorder(payload: Any, ctx: Any) -> Any:
            observed.append(payload)
            return None

        register(recorder)

        async def _impl(action, context, backend, requirements, **kwargs):
            return _make_minimal_sampling_result(context)

        strategy = _MinimalStrategy(_impl, loop_budget=1)
        await strategy.sample(
            Instruction("Wrapper test"),
            context=SimpleContext(),
            backend=_MockBackend(),
            requirements=[],
            show_progress=False,
        )

        assert len(observed) == 1
        assert "Minimal" in observed[0].strategy_name

    async def test_loop_end_fires_from_wrapper_on_success(self) -> None:
        """sampling_loop_end fires from the wrapper on the success path."""
        observed: list[Any] = []

        @hook("sampling_loop_end")
        async def recorder(payload: Any, ctx: Any) -> Any:
            observed.append(payload)
            return None

        register(recorder)

        async def _impl(action, context, backend, requirements, **kwargs):
            return _make_minimal_sampling_result(context)

        strategy = _MinimalStrategy(_impl, loop_budget=1)
        await strategy.sample(
            Instruction("End-hook test"),
            context=SimpleContext(),
            backend=_MockBackend(),
            requirements=[],
            show_progress=False,
        )

        assert len(observed) == 1
        assert observed[0].success is True

    async def test_loop_end_fires_on_exception_path(self) -> None:
        """sampling_loop_end fires from the wrapper even when _sample_impl raises."""
        observed: list[Any] = []

        @hook("sampling_loop_end")
        async def recorder(payload: Any, ctx: Any) -> Any:
            observed.append(payload)
            return None

        register(recorder)

        async def _raising_impl(action, context, backend, requirements, **kwargs):
            raise RuntimeError("impl boom")

        strategy = _MinimalStrategy(_raising_impl, loop_budget=1)

        with pytest.raises(RuntimeError, match="impl boom"):
            await strategy.sample(
                Instruction("Exception-path test"),
                context=SimpleContext(),
                backend=_MockBackend(),
                requirements=[],
                show_progress=False,
            )

        assert len(observed) == 1
        assert observed[0].exception is not None
        assert isinstance(observed[0].exception, RuntimeError)
        assert observed[0].success is False

    async def test_loop_end_fires_when_start_hook_raises(self) -> None:
        """sampling_loop_end fires even when invoke_hook for sampling_loop_start raises.

        The framework swallows ordinary plugin exceptions, but PluginViolationError
        (and any other future leak from invoke_hook) would previously escape before
        entering the try/finally that fires sampling_loop_end.  The fix moves the
        start-hook call inside the protected region, so sampling_loop_end always fires.
        """
        from mellea.plugins import PluginViolationError
        from mellea.plugins.types import HookType as _HookType

        end_payloads: list[Any] = []

        @hook("sampling_loop_end")
        async def end_recorder(payload: Any, ctx: Any) -> Any:
            end_payloads.append(payload)
            return None

        register(end_recorder)

        async def _unreachable_impl(action, context, backend, requirements, **kwargs):
            raise AssertionError("should not reach _sample_impl")

        strategy = _MinimalStrategy(_unreachable_impl, loop_budget=1)

        boom = PluginViolationError(
            hook_type="sampling_loop_start",
            reason="start hook boom",
            code="TEST",
            plugin_name="test_plugin",
        )

        # sample() does `from ..plugins.manager import has_plugins, invoke_hook`
        # inside the function body, so patching the source module attributes
        # affects those local references.
        #
        # has_plugins is patched to return True for both hook types so that both
        # the SAMPLING_LOOP_START block and the SAMPLING_LOOP_END finally block
        # are entered.  invoke_hook raises PluginViolationError only for
        # SAMPLING_LOOP_START; for SAMPLING_LOOP_END it delegates to the real
        # implementation, which dispatches to end_recorder above.
        import mellea.plugins.manager as _mgr

        real_invoke_hook = _mgr.invoke_hook

        async def _start_raises(hook_type, payload, **kwargs):
            if hook_type is _HookType.SAMPLING_LOOP_START:
                raise boom
            return await real_invoke_hook(hook_type, payload, **kwargs)

        with (
            patch("mellea.plugins.manager.has_plugins", side_effect=lambda _ht: True),
            patch("mellea.plugins.manager.invoke_hook", side_effect=_start_raises),
            pytest.raises(PluginViolationError, match="start hook boom"),
        ):
            await strategy.sample(
                Instruction("Start-hook exception test"),
                context=SimpleContext(),
                backend=_MockBackend(),
                requirements=[],
                show_progress=False,
            )

        assert len(end_payloads) == 1
        assert end_payloads[0].success is False
        assert isinstance(end_payloads[0].exception, PluginViolationError)
        assert "start hook boom" in str(end_payloads[0].exception)

    async def test_loop_end_fires_when_merge_requirements_raises(self) -> None:
        """sampling_loop_end fires even when _merge_requirements raises before the try block.

        Previously, _merge_requirements() executed outside the try/finally, so any
        exception there would escape without firing sampling_loop_end.  The fix moves
        setup inside the protected region so the end-hook closure is guaranteed.
        """
        end_payloads: list[Any] = []

        @hook("sampling_loop_end")
        async def end_recorder(payload: Any, ctx: Any) -> Any:
            end_payloads.append(payload)
            return None

        register(end_recorder)

        async def _unreachable_impl(action, context, backend, requirements, **kwargs):
            raise AssertionError("should not reach _sample_impl")

        strategy = _MinimalStrategy(_unreachable_impl, loop_budget=1)

        boom = RuntimeError("merge requirements boom")
        with (
            patch.object(strategy, "_merge_requirements", side_effect=boom),
            pytest.raises(RuntimeError, match="merge requirements boom"),
        ):
            await strategy.sample(
                Instruction("Merge-requirements exception test"),
                context=SimpleContext(),
                backend=_MockBackend(),
                requirements=[],
                show_progress=False,
            )

        assert len(end_payloads) == 1
        assert end_payloads[0].success is False
        assert end_payloads[0].exception is boom

    async def test_sampling_id_correlates_start_and_end(self) -> None:
        """sampling_loop_start and sampling_loop_end carry the same sampling_id."""
        start_ids: list[str] = []
        end_ids: list[str] = []

        @hook("sampling_loop_start")
        async def start_rec(payload: Any, ctx: Any) -> Any:
            start_ids.append(payload.sampling_id)
            return None

        @hook("sampling_loop_end")
        async def end_rec(payload: Any, ctx: Any) -> Any:
            end_ids.append(payload.sampling_id)
            return None

        register(start_rec)
        register(end_rec)

        strategy = RejectionSamplingStrategy(loop_budget=1)
        await strategy.sample(
            Instruction("ID correlation test"),
            context=SimpleContext(),
            backend=_MockBackend(),
            requirements=[],
            show_progress=False,
        )

        assert len(start_ids) == 1
        assert len(end_ids) == 1
        assert start_ids[0] == end_ids[0]

    async def test_start_hook_modified_budget_reaches_sample_impl(self) -> None:
        """A start hook that raises loop_budget is reflected in _sample_impl's effective_loop_budget.

        The wrapper reads `start_payload.loop_budget` after the hook returns and
        forwards it as `effective_loop_budget` to `_sample_impl`.  This test
        directly verifies that contract by inspecting the kwargs received by the
        impl rather than inferring it through a full strategy run.

        Also checks that the `sampling_id` seen by `_sample_impl` matches the one
        emitted on the `sampling_loop_start` payload.
        """
        captured_start_id: list[str] = []
        captured_impl_kwargs: list[dict] = []

        @hook("sampling_loop_start")
        async def bump_budget(payload: Any, ctx: Any) -> Any:
            captured_start_id.append(payload.sampling_id)
            return PluginResult(
                continue_processing=True,
                modified_payload=payload.model_copy(update={"loop_budget": 3}),
            )

        register(bump_budget)

        async def _impl(action, context, backend, requirements, **kwargs):
            captured_impl_kwargs.append(dict(kwargs))
            return _make_minimal_sampling_result(context)

        strategy = _MinimalStrategy(_impl, loop_budget=1)
        await strategy.sample(
            Instruction("Modified budget test"),
            context=SimpleContext(),
            backend=_MockBackend(),
            requirements=[],
            show_progress=False,
        )

        assert len(captured_start_id) == 1
        assert len(captured_impl_kwargs) == 1

        # The hook raised the budget from 1 to 3; _sample_impl must see 3.
        assert captured_impl_kwargs[0]["effective_loop_budget"] == 3

        # The sampling_id forwarded to _sample_impl must match what the start hook saw.
        assert captured_impl_kwargs[0]["sampling_id"] == captured_start_id[0]

    async def test_zero_budget_from_start_hook_closes_lifecycle(self) -> None:
        """A start hook returning loop_budget=0 causes ValueError before _sample_impl.

        The wrapper validates the hook-modified budget and raises ValueError for
        any value < 1.  This test asserts that in that failure path:
          - _sample_impl is never invoked,
          - sample() raises ValueError,
          - sampling_loop_end still fires (lifecycle is closed),
          - the end payload carries success=False,
          - the end payload exception is the same ValueError,
          - the end payload sampling_id matches the one on the start payload.
        """
        captured_start_ids: list[str] = []
        end_payloads: list[Any] = []
        impl_called: list[bool] = []

        @hook("sampling_loop_start")
        async def zero_budget(payload: Any, ctx: Any) -> Any:
            captured_start_ids.append(payload.sampling_id)
            return PluginResult(
                continue_processing=True,
                modified_payload=payload.model_copy(update={"loop_budget": 0}),
            )

        @hook("sampling_loop_end")
        async def end_rec(payload: Any, ctx: Any) -> Any:
            end_payloads.append(payload)
            return None

        register(zero_budget)
        register(end_rec)

        async def _impl(action, context, backend, requirements, **kwargs):
            impl_called.append(True)
            return _make_minimal_sampling_result(context)

        strategy = _MinimalStrategy(_impl, loop_budget=1)

        with pytest.raises(ValueError, match="non-positive loop_budget"):
            await strategy.sample(
                Instruction("Zero budget test"),
                context=SimpleContext(),
                backend=_MockBackend(),
                requirements=[],
                show_progress=False,
            )

        # _sample_impl must never have been reached.
        assert impl_called == []

        # sampling_loop_end must still have fired once.
        assert len(end_payloads) == 1
        end = end_payloads[0]
        assert end.success is False
        assert isinstance(end.exception, ValueError)
        assert "non-positive loop_budget" in str(end.exception)

        # The end payload must use the same sampling_id as the start payload.
        assert len(captured_start_ids) == 1
        assert end.sampling_id == captured_start_ids[0]


# ---------------------------------------------------------------------------
# Session hook call sites
# ---------------------------------------------------------------------------


class TestSessionHookCallSites:
    """SESSION_PRE_INIT and SESSION_POST_INIT fire in start_session().

    start_session() is a synchronous function that uses _run_async_in_thread
    to invoke hooks.  These tests patch backend_name_to_class to avoid
    instantiating a real LLM backend.
    """

    def test_session_pre_init_fires_during_start_session(self) -> None:
        """SESSION_PRE_INIT fires once before the backend is instantiated."""
        from mellea.stdlib.session import start_session

        observed: list[Any] = []

        @hook("session_pre_init")
        async def recorder(payload: Any, ctx: Any) -> Any:
            observed.append(payload)
            return None

        register(recorder)

        with patch(
            "mellea.stdlib.session.backend_name_to_class", return_value=_MockBackend
        ):
            start_session("ollama", model_id="test-model")

        assert len(observed) == 1

    def test_session_pre_init_payload_has_backend_name_and_model_id(self) -> None:
        """SESSION_PRE_INIT payload carries the backend_name and model_id passed to start_session."""
        from mellea.stdlib.session import start_session

        observed: list[Any] = []

        @hook("session_pre_init")
        async def recorder(payload: Any, ctx: Any) -> Any:
            observed.append(payload)
            return None

        register(recorder)

        with patch(
            "mellea.stdlib.session.backend_name_to_class", return_value=_MockBackend
        ):
            start_session("ollama", model_id="granite-3b-instruct")

        p = observed[0]
        assert p.backend_name == "ollama"
        assert p.model_id == "granite-3b-instruct"

    def test_session_post_init_fires_after_session_created(self) -> None:
        """SESSION_POST_INIT fires once after the MelleaSession object is created."""
        from mellea.stdlib.session import start_session

        observed: list[Any] = []

        @hook("session_post_init")
        async def recorder(payload: Any, ctx: Any) -> Any:
            observed.append(payload)
            return None

        register(recorder)

        with patch(
            "mellea.stdlib.session.backend_name_to_class", return_value=_MockBackend
        ):
            start_session("ollama", model_id="test-model")

        assert len(observed) == 1

    def test_session_post_init_payload_has_session_metadata(self) -> None:
        """SESSION_POST_INIT payload contains session_id, model_id, and context."""
        from mellea.stdlib.session import start_session

        observed: list[Any] = []

        @hook("session_post_init")
        async def recorder(payload: Any, ctx: Any) -> Any:
            observed.append(payload)
            return None

        register(recorder)

        with patch(
            "mellea.stdlib.session.backend_name_to_class", return_value=_MockBackend
        ):
            session = start_session("ollama", model_id="test-model")

        p = observed[0]
        assert p.session_id == session.id
        assert p.model_id == "test-model"
        assert p.context is not None

    def test_pre_init_fires_before_post_init(self) -> None:
        """SESSION_PRE_INIT fires before SESSION_POST_INIT."""
        from mellea.stdlib.session import start_session

        order: list[str] = []

        @hook("session_pre_init")
        async def pre_recorder(payload: Any, ctx: Any) -> Any:
            order.append("pre_init")
            return None

        @hook("session_post_init")
        async def post_recorder(payload: Any, ctx: Any) -> Any:
            order.append("post_init")
            return None

        register(pre_recorder)
        register(post_recorder)

        with patch(
            "mellea.stdlib.session.backend_name_to_class", return_value=_MockBackend
        ):
            start_session("ollama", model_id="test-model")

        assert order == ["pre_init", "post_init"]

    def test_session_pre_init_mutation_is_applied_before_backend_init(self) -> None:
        """SESSION_PRE_INIT mutations reach the backend constructor."""
        from mellea.stdlib.session import start_session

        order: list[str] = []
        captured_backend_args: dict[str, Any] = {}

        class RecordingBackend(_MockBackend):
            def __init__(self, model_id, model_options=None, **kwargs):
                order.append("backend_init")
                captured_backend_args["model_id"] = model_id
                captured_backend_args["model_options"] = model_options
                captured_backend_args["kwargs"] = kwargs

        async def fake_invoke_hook(hook_type, payload, **_kwargs):
            assert hook_type is HookType.SESSION_PRE_INIT
            order.append("hook")
            modified = payload.model_copy(
                update={
                    "model_id": "hook-model",
                    "model_options": {"temperature": 0.25},
                }
            )
            return (
                PluginResult(continue_processing=True, modified_payload=modified),
                modified,
            )

        def has_session_pre_init(hook_type=None):
            return hook_type is HookType.SESSION_PRE_INIT

        with (
            patch(
                "mellea.stdlib.session.has_plugins", side_effect=has_session_pre_init
            ),
            patch("mellea.stdlib.session.invoke_hook", side_effect=fake_invoke_hook),
            patch(
                "mellea.stdlib.session.backend_name_to_class",
                return_value=RecordingBackend,
            ),
        ):
            start_session(
                "ollama", model_id="original-model", model_options={"temperature": 1.0}
            )

        assert order == ["hook", "backend_init"]
        assert captured_backend_args["model_id"] == "hook-model"
        assert captured_backend_args["model_options"] == {"temperature": 0.25}


# ---------------------------------------------------------------------------
# Mutation tests — verify that hook-modified payloads are actually applied
# ---------------------------------------------------------------------------


class TestGenerationPostCallObserveOnly:
    """GENERATION_POST_CALL is observe-only — modifications are discarded."""

    async def test_modification_discarded_on_eager_path(self) -> None:
        """A plugin that tries to replace model_output has its change discarded."""
        replacement = MagicMock(spec=ModelOutputThunk)
        replacement._generate_log = None

        @hook("generation_post_call")
        async def swap_output(payload, *_):
            modified = payload.model_copy(update={"model_output": replacement})
            return PluginResult(continue_processing=True, modified_payload=modified)

        register(swap_output)
        backend = _MockBackend()
        result, _ = await backend.generate_from_context(
            CBlock("mutation test"), MagicMock(spec=Context)
        )

        # model_output is no longer writable — original is preserved
        assert result is not replacement
        assert isinstance(result, ModelOutputThunk)

    async def test_no_modification_returns_original_output(self) -> None:
        """When the hook returns None the original thunk is returned unchanged."""

        @hook("generation_post_call")
        async def observe_only(*_):
            return None

        register(observe_only)
        backend = _MockBackend()
        result, _ = await backend.generate_from_context(
            CBlock("no-op test"), MagicMock(spec=Context)
        )

        assert result is not None
        assert isinstance(result, ModelOutputThunk)


# ---------------------------------------------------------------------------
# Lazy/stream path MOT replacement
# ---------------------------------------------------------------------------


class _MockLazyBackend(Backend):
    """Backend that returns a real lazy (uncomputed) ModelOutputThunk.

    The MOT must be materialized via `avalue()`/`astream()`, exercising
    the `_on_computed` callback path.
    """

    model_id = "mock-lazy-model"

    def __init__(self, *args, **kwargs):
        self._model_id: str = "mock-lazy-model"
        self._provider: str = "mock-provider"

    async def _generate_from_context(self, action, ctx, **kwargs):
        import asyncio

        mot = ModelOutputThunk(value=None)
        mot._gen.generate_type = GenerateType.ASYNC
        mot._gen.chunk_size = 0
        mot._call.action = action

        async def _process(thunk, chunk):
            if thunk._underlying_value is None:
                thunk._underlying_value = ""
            thunk._underlying_value += str(chunk)

        async def _post_process(thunk):
            pass

        mot._gen.process = _process
        mot._gen.post_process = _post_process

        glog = GenerateLog()
        glog.prompt = "lazy mocked prompt"
        mot._generate_log = glog

        # Simulate async generation: enqueue chunks + sentinel
        async def _generate():
            await mot._gen.queue.put("lazy output")
            await mot._gen.queue.put(None)  # sentinel

        mot._gen.generate = asyncio.ensure_future(_generate())

        return mot, SimpleContext()

    async def _generate_from_raw(self, actions, ctx, **kwargs):
        return [], None


class TestGenerationPostCallObserveOnlyLazyPath:
    """GENERATION_POST_CALL is observe-only on the lazy/stream path."""

    async def test_modification_discarded_on_lazy_path(self) -> None:
        """A plugin trying to replace model_output has its change discarded on lazy path."""
        replacement = ModelOutputThunk(value="replaced output")
        replacement_glog = GenerateLog()
        replacement_glog.prompt = "replaced prompt"
        replacement._generate_log = replacement_glog

        @hook("generation_post_call")
        async def swap_output(payload, *_):
            modified = payload.model_copy(update={"model_output": replacement})
            return PluginResult(continue_processing=True, modified_payload=modified)

        register(swap_output)
        backend = _MockLazyBackend()
        result, _ = await backend.generate_from_context(
            CBlock("lazy mutation test"), MagicMock(spec=Context)
        )

        # model_output is no longer writable — original value is preserved
        await result.avalue()
        assert result.value == "lazy output"

    async def test_no_modification_preserves_original_on_lazy_path(self) -> None:
        """When the hook returns None on the lazy path, the original MOT is unchanged."""

        @hook("generation_post_call")
        async def observe_only(*_):
            return None

        register(observe_only)
        backend = _MockLazyBackend()
        result, _ = await backend.generate_from_context(
            CBlock("lazy no-op test"), MagicMock(spec=Context)
        )

        value = await result.avalue()
        assert value == "lazy output"

    async def test_hook_fires_exactly_once_on_lazy_path(self) -> None:
        """GENERATION_POST_CALL fires exactly once even when avalue() is called after astream()."""
        fire_count = 0

        @hook("generation_post_call")
        async def counter(*_):
            nonlocal fire_count
            fire_count += 1
            return None

        register(counter)
        backend = _MockLazyBackend()
        result, _ = await backend.generate_from_context(
            CBlock("lazy fire-once test"), MagicMock(spec=Context)
        )

        await result.avalue()
        # Second avalue call should not re-fire
        await result.avalue()
        assert fire_count == 1


class TestSamplingLoopEndObserveOnly:
    """SAMPLING_LOOP_END is observe-only — modifications are discarded."""

    async def test_observe_only_on_success_path(self) -> None:
        """Hook fires on success but cannot modify final_result."""
        from mellea.stdlib.components import Instruction
        from mellea.stdlib.sampling.base import RejectionSamplingStrategy

        observed: list[bool] = []

        @hook("sampling_loop_end")
        async def observe_success(payload, *_):
            observed.append(payload.success)
            return None

        register(observe_success)
        backend = _MockBackend()
        ctx = SimpleContext()
        strategy = RejectionSamplingStrategy(loop_budget=1)

        sampling_result = await strategy.sample(
            Instruction("observe test"),
            context=ctx,
            backend=backend,
            requirements=[],
            format=None,
            model_options=None,
            tool_calls=False,
            show_progress=False,
        )

        assert sampling_result.result is not None
        assert isinstance(sampling_result.result, ModelOutputThunk)
        assert observed == [True]

    async def test_observe_only_on_failure_path(self) -> None:
        """Hook fires on failure but cannot modify final_result."""
        from mellea.core.requirement import Requirement, ValidationResult
        from mellea.stdlib.components import Instruction
        from mellea.stdlib.sampling.base import RejectionSamplingStrategy

        observed: list[bool] = []

        @hook("sampling_loop_end")
        async def observe_failure(payload, *_):
            observed.append(payload.success)
            return None

        register(observe_failure)
        backend = _MockBackend()
        ctx = SimpleContext()

        always_fail = Requirement(
            description="always fails",
            validation_fn=lambda _ctx: ValidationResult(
                result=False, reason="forced failure"
            ),
        )
        strategy = RejectionSamplingStrategy(loop_budget=1)

        sampling_result = await strategy.sample(
            Instruction("failure observe test"),
            context=ctx,
            backend=backend,
            requirements=[always_fail],
            format=None,
            model_options=None,
            tool_calls=False,
            show_progress=False,
        )

        assert not sampling_result.success
        assert sampling_result.result is not None
        assert observed == [False]


# ---------------------------------------------------------------------------
# Streaming hook call sites
# ---------------------------------------------------------------------------


async def _feed_tokens(mot: ModelOutputThunk, response: str) -> None:
    for ch in response:
        await mot._gen.queue.put(ch)
        await asyncio.sleep(0)
    await mot._gen.queue.put(None)


class _StreamingBackend(Backend):
    """Streams a fixed response one character at a time via an async MOT."""

    def __init__(self, response: str = "Hello world. ") -> None:
        self._response = response
        self._model_id = "stream-mock-model"
        self._provider = "stream-mock-provider"

    async def _generate_from_context(self, action, ctx, **kwargs):
        mot = _make_thunk()
        mot.generation.model = self._model_id
        mot.generation.provider = self._provider
        task = asyncio.create_task(_feed_tokens(mot, self._response))
        _ = task
        return mot, ctx.add(action).add(mot)

    async def _generate_from_raw(self, actions, ctx, **kwargs):
        raise NotImplementedError


class TestStreamingHookCallSites:
    """STREAMING_START/EVENT/END fire across a stream() run consumed with async for."""

    async def test_start_and_end_fire_once_across_a_drained_stream(self) -> None:
        """Draining a stream fires STREAMING_START and STREAMING_END once each, with payloads."""
        from mellea.stdlib.streaming import stream

        starts: list[Any] = []
        ends: list[Any] = []

        @hook("streaming_start")
        async def on_start(payload: Any, ctx: Any) -> Any:
            starts.append(payload)
            return None

        @hook("streaming_end")
        async def on_end(payload: Any, ctx: Any) -> Any:
            ends.append(payload)
            return None

        register(on_start)
        register(on_end)
        async with await stream(
            CBlock("prompt"), _StreamingBackend(), SimpleContext(), chunking="sentence"
        ) as s:
            async for _chunk in s:
                pass

        assert len(starts) == 1
        assert starts[0].has_requirements is False
        assert starts[0].requirement_count == 0
        assert starts[0].chunking_strategy == "SentenceChunking"

        assert len(ends) == 1
        assert ends[0].success is True
        assert ends[0].model == "stream-mock-model"
        assert ends[0].provider == "stream-mock-provider"

    async def test_streaming_end_fires_once_across_repeat_aclose(self) -> None:
        """aclose() fires STREAMING_END once; a repeat aclose() does not re-fire it."""
        from mellea.stdlib.streaming import stream

        observed: list[Any] = []

        @hook("streaming_end")
        async def recorder(payload: Any, ctx: Any) -> Any:
            observed.append(payload)
            return None

        register(recorder)
        # Abnormal path on purpose: the Streamer is never consumed, so aclose()
        # fires END rather than a natural drain.
        s = await stream(CBlock("prompt"), _StreamingBackend(), SimpleContext())
        await s.aclose()
        await s.aclose()

        assert len(observed) == 1

    async def test_streaming_end_fires_when_generation_raises(self) -> None:
        """A backend failure during setup fires STREAMING_END with no model."""
        from mellea.stdlib.streaming import stream

        class _RaisingBackend(Backend):
            _model_id = "x"
            _provider = "y"

            async def _generate_from_context(self, action, ctx, **kwargs):
                raise RuntimeError("backend down")

            async def _generate_from_raw(self, actions, ctx, **kwargs):
                raise NotImplementedError

        observed: list[Any] = []

        @hook("streaming_end")
        async def recorder(payload: Any, ctx: Any) -> Any:
            observed.append(payload)
            return None

        register(recorder)
        with pytest.raises(RuntimeError, match="backend down"):
            await stream(CBlock("prompt"), _RaisingBackend(), SimpleContext())

        assert len(observed) == 1
        assert observed[0].success is False
        assert observed[0].model is None
        assert observed[0].provider is None


if __name__ == "__main__":
    pytest.main([__file__])
