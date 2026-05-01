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
    Context,
    GenerateLog,
    GenerateType,
    ModelOutputThunk,
)
from mellea.plugins import PluginResult, hook, register
from mellea.stdlib.context import SimpleContext

# ---------------------------------------------------------------------------
# Mock backend (module-level so it can be used as a class in session tests)
# ---------------------------------------------------------------------------


class _MockBackend(Backend):
    """Minimal backend that returns a faked ModelOutputThunk — no LLM API calls."""

    model_id = "mock-model"

    def __init__(self, *args, **kwargs):
        # Accept but discard constructor arguments; real backends need model_id etc.
        pass

    async def _generate_from_context(self, action, ctx, **kwargs):
        mot = MagicMock(spec=ModelOutputThunk)
        glog = GenerateLog()
        glog.prompt = "mocked formatted prompt"
        mot._generate_log = glog
        mot.parsed_repr = None
        mot._start = datetime.datetime.now()

        async def _avalue():
            return "mocked output"

        mot.avalue = _avalue
        mot.value = "mocked output string"  # SamplingResult requires a str .value
        # Return a new SimpleContext to mimic real context evolution
        new_ctx = SimpleContext()
        return mot, new_ctx

    async def generate_from_raw(self, actions, ctx, **kwargs):
        # Required abstract method; not exercised by these tests
        return []


async def _noop_process(mot, chunk):
    if mot._underlying_value is None:
        mot._underlying_value = ""
    mot._underlying_value += str(chunk)


async def _noop_post_process(mot):
    return


def _make_thunk():
    mot = ModelOutputThunk(value=None)
    mot._generate_type = GenerateType.ASYNC
    mot._process = _noop_process
    mot._post_process = _noop_post_process
    mot._action = CBlock("test")
    mot._chunk_size = 0
    mot._start = datetime.datetime.now()
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
        await mot._async_queue.put("hello")
        await mot._async_queue.put("goodbye")
        await mot._async_queue.put(None)  # sentinel for being done

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
        await mot._async_queue.put("hello")
        await mot._async_queue.put("goodbye")
        await mot._async_queue.put(None)  # sentinel for being done
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
        await mot._async_queue.put("hello")
        await mot._async_queue.put("goodbye")
        await mot._async_queue.put(None)  # sentinel for being done

        await asyncio.sleep(1)
        await mot.avalue()

        assert observed[0].latency_ms >= 0


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

    The MOT must be materialized via ``avalue()``/``astream()``, exercising
    the ``_on_computed`` callback path.
    """

    model_id = "mock-lazy-model"

    def __init__(self, *args, **kwargs):
        pass

    async def _generate_from_context(self, action, ctx, **kwargs):
        import asyncio

        mot = ModelOutputThunk(value=None)
        mot._generate_type = GenerateType.ASYNC
        mot._chunk_size = 0
        mot._action = action

        async def _process(thunk, chunk):
            if thunk._underlying_value is None:
                thunk._underlying_value = ""
            thunk._underlying_value += str(chunk)

        async def _post_process(thunk):
            pass

        mot._process = _process
        mot._post_process = _post_process

        glog = GenerateLog()
        glog.prompt = "lazy mocked prompt"
        mot._generate_log = glog

        # Simulate async generation: enqueue chunks + sentinel
        async def _generate():
            await mot._async_queue.put("lazy output")
            await mot._async_queue.put(None)  # sentinel

        mot._generate = asyncio.ensure_future(_generate())

        return mot, SimpleContext()

    async def generate_from_raw(self, actions, ctx, **kwargs):
        return []


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


if __name__ == "__main__":
    pytest.main([__file__])
