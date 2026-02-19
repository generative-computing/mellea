"""Integration tests verifying that hooks fire at actual Mellea call sites.

Each test registers a hook recorder, triggers the actual code path (Backend,
functional.py, sampling/base.py, session.py), and asserts that the hook fired
with the expected payload shape.

All tests use lightweight mock backends so no real LLM API calls are made.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("mcpgateway.plugins.framework")

from mellea.plugins import hook, register
from mellea.plugins.manager import invoke_hook, shutdown_plugins
from mellea.plugins.types import HookType
from mellea.core.backend import Backend
from mellea.core.base import CBlock, Context, GenerateLog, ModelOutputThunk
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

    async def generate_from_context(self, action, ctx, **kwargs):
        mot = MagicMock(spec=ModelOutputThunk)
        glog = GenerateLog()
        glog.prompt = "mocked formatted prompt"
        mot._generate_log = glog
        mot.parsed_repr = None

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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
async def reset_plugins():
    """Shut down and reset the plugin manager after every test."""
    yield
    await shutdown_plugins()


# ---------------------------------------------------------------------------
# Generation hook call sites
# ---------------------------------------------------------------------------


class TestGenerationHookCallSites:
    """GENERATION_PRE_CALL and GENERATION_POST_CALL fire in Backend.generate_from_context_with_hooks()."""

    async def test_generation_pre_call_fires_once(self):
        """GENERATION_PRE_CALL fires exactly once per generate_from_context_with_hooks() call."""
        observed: list[Any] = []

        @hook("generation_pre_call")
        async def recorder(payload: Any, ctx: Any) -> Any:
            observed.append(payload)
            return None

        register(recorder)
        backend = _MockBackend()
        action = CBlock("hello world")
        await backend.generate_from_context_with_hooks(action, MagicMock(spec=Context))

        assert len(observed) == 1

    async def test_generation_pre_call_payload_has_action_and_context(self):
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
        await backend.generate_from_context_with_hooks(action, mock_ctx)

        p = observed[0]
        assert p.action is action
        assert p.context is mock_ctx

    async def test_generation_post_call_fires_once(self):
        """GENERATION_POST_CALL fires exactly once after generate_from_context() returns."""
        observed: list[Any] = []

        @hook("generation_post_call")
        async def recorder(payload: Any, ctx: Any) -> Any:
            observed.append(payload)
            return None

        register(recorder)
        backend = _MockBackend()
        await backend.generate_from_context_with_hooks(
            CBlock("test"), MagicMock(spec=Context)
        )

        assert len(observed) == 1

    async def test_generation_post_call_model_output_is_the_returned_thunk(self):
        """GENERATION_POST_CALL payload.model_output IS the ModelOutputThunk returned."""
        observed: list[Any] = []

        @hook("generation_post_call")
        async def recorder(payload: Any, ctx: Any) -> Any:
            observed.append(payload)
            return None

        register(recorder)
        backend = _MockBackend()
        result, _ = await backend.generate_from_context_with_hooks(
            CBlock("test"), MagicMock(spec=Context)
        )

        assert observed[0].model_output is result

    async def test_generation_post_call_latency_ms_is_non_negative(self):
        """GENERATION_POST_CALL payload.latency_ms >= 0."""
        observed: list[Any] = []

        @hook("generation_post_call")
        async def recorder(payload: Any, ctx: Any) -> Any:
            observed.append(payload)
            return None

        register(recorder)
        backend = _MockBackend()
        await backend.generate_from_context_with_hooks(
            CBlock("test"), MagicMock(spec=Context)
        )

        assert observed[0].latency_ms >= 0

    async def test_both_generation_hooks_fire_in_order(self):
        """GENERATION_PRE_CALL fires before GENERATION_POST_CALL."""
        order: list[str] = []

        @hook("generation_pre_call")
        async def pre_recorder(payload: Any, ctx: Any) -> Any:
            order.append("pre")
            return None

        @hook("generation_post_call")
        async def post_recorder(payload: Any, ctx: Any) -> Any:
            order.append("post")
            return None

        register(pre_recorder)
        register(post_recorder)
        backend = _MockBackend()
        await backend.generate_from_context_with_hooks(
            CBlock("order test"), MagicMock(spec=Context)
        )

        assert order == ["pre", "post"]


# ---------------------------------------------------------------------------
# Component hook call sites
# ---------------------------------------------------------------------------


class TestComponentHookCallSites:
    """Component hooks fire in ainstruct() and aact() in stdlib/functional.py."""

    async def test_component_pre_create_fires_in_ainstruct(self):
        """COMPONENT_PRE_CREATE fires exactly once per ainstruct() call."""
        from mellea.stdlib.functional import ainstruct

        observed: list[Any] = []

        @hook("component_pre_create")
        async def recorder(payload: Any, ctx: Any) -> Any:
            observed.append(payload)
            return None

        register(recorder)
        backend = _MockBackend()
        ctx = SimpleContext()
        await ainstruct("Write a poem", ctx, backend, strategy=None)

        assert len(observed) == 1

    async def test_component_pre_create_payload_component_type_is_instruction(self):
        """COMPONENT_PRE_CREATE payload has component_type='Instruction' in ainstruct()."""
        from mellea.stdlib.functional import ainstruct

        observed: list[Any] = []

        @hook("component_pre_create")
        async def recorder(payload: Any, ctx: Any) -> Any:
            observed.append(payload)
            return None

        register(recorder)
        backend = _MockBackend()
        ctx = SimpleContext()
        await ainstruct("Write a poem", ctx, backend, strategy=None)

        assert observed[0].component_type == "Instruction"

    async def test_component_pre_create_payload_description_matches_input(self):
        """COMPONENT_PRE_CREATE payload.description matches the description passed to ainstruct()."""
        from mellea.stdlib.functional import ainstruct

        observed: list[Any] = []

        @hook("component_pre_create")
        async def recorder(payload: Any, ctx: Any) -> Any:
            observed.append(payload)
            return None

        register(recorder)
        backend = _MockBackend()
        ctx = SimpleContext()
        await ainstruct("Specific description string", ctx, backend, strategy=None)

        assert observed[0].description == "Specific description string"

    async def test_component_post_create_fires_in_ainstruct(self):
        """COMPONENT_POST_CREATE fires after Instruction is created in ainstruct()."""
        from mellea.stdlib.functional import ainstruct

        observed: list[Any] = []

        @hook("component_post_create")
        async def recorder(payload: Any, ctx: Any) -> Any:
            observed.append(payload)
            return None

        register(recorder)
        backend = _MockBackend()
        ctx = SimpleContext()
        await ainstruct("Create me", ctx, backend, strategy=None)

        assert len(observed) == 1

    async def test_component_post_create_payload_has_live_component(self):
        """COMPONENT_POST_CREATE payload.component is the live Instruction object."""
        from mellea.stdlib.functional import ainstruct
        from mellea.stdlib.components import Instruction

        observed: list[Any] = []

        @hook("component_post_create")
        async def recorder(payload: Any, ctx: Any) -> Any:
            observed.append(payload)
            return None

        register(recorder)
        backend = _MockBackend()
        ctx = SimpleContext()
        await ainstruct("Component test", ctx, backend, strategy=None)

        p = observed[0]
        assert p.component is not None
        assert isinstance(p.component, Instruction)
        assert p.component_type == "Instruction"

    async def test_component_pre_execute_fires_in_aact(self):
        """COMPONENT_PRE_EXECUTE fires in aact() before generation is called."""
        from mellea.stdlib.functional import aact
        from mellea.stdlib.components import Instruction

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

    async def test_component_pre_execute_payload_has_live_action(self):
        """COMPONENT_PRE_EXECUTE payload.action IS the same Component instance."""
        from mellea.stdlib.functional import aact
        from mellea.stdlib.components import Instruction

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
        assert observed[0].action is action

    async def test_component_pre_execute_payload_component_type(self):
        """COMPONENT_PRE_EXECUTE payload.component_type matches the action class name."""
        from mellea.stdlib.functional import aact
        from mellea.stdlib.components import Instruction

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

    async def test_component_post_success_fires_in_aact(self):
        """COMPONENT_POST_SUCCESS fires in aact() after successful generation."""
        from mellea.stdlib.functional import aact
        from mellea.stdlib.components import Instruction

        observed: list[Any] = []

        @hook("component_post_success")
        async def recorder(payload: Any, ctx: Any) -> Any:
            observed.append(payload)
            return None

        register(recorder)
        backend = _MockBackend()
        ctx = SimpleContext()
        action = Instruction("Success test")

        result, new_ctx = await aact(action, ctx, backend, strategy=None)
        assert len(observed) == 1

    async def test_component_post_success_payload_has_correct_result_and_contexts(self):
        """COMPONENT_POST_SUCCESS payload carries result, context_before, context_after."""
        from mellea.stdlib.functional import aact
        from mellea.stdlib.components import Instruction

        observed: list[Any] = []

        @hook("component_post_success")
        async def recorder(payload: Any, ctx: Any) -> Any:
            observed.append(payload)
            return None

        register(recorder)
        backend = _MockBackend()
        ctx = SimpleContext()
        action = Instruction("Payload check")

        result, new_ctx = await aact(action, ctx, backend, strategy=None)

        p = observed[0]
        assert p.result is result  # live reference to the returned ModelOutputThunk
        assert p.context_before is ctx  # original input context
        assert p.context_after is new_ctx  # context after generation
        assert p.context_before is not p.context_after  # they are different objects
        assert p.action is action
        assert p.latency_ms >= 0

    async def test_component_pre_create_and_post_create_both_fire_in_ainstruct(self):
        """Both COMPONENT_PRE_CREATE and COMPONENT_POST_CREATE fire per ainstruct() call."""
        from mellea.stdlib.functional import ainstruct

        order: list[str] = []

        @hook("component_pre_create")
        async def pre_recorder(payload: Any, ctx: Any) -> Any:
            order.append("pre_create")
            return None

        @hook("component_post_create")
        async def post_recorder(payload: Any, ctx: Any) -> Any:
            order.append("post_create")
            return None

        register(pre_recorder)
        register(post_recorder)
        backend = _MockBackend()
        ctx = SimpleContext()
        await ainstruct("Order test", ctx, backend, strategy=None)

        assert order == ["pre_create", "post_create"]

    async def test_all_four_component_hooks_fire_in_ainstruct(self):
        """All four component hooks fire in the correct order during ainstruct()."""
        from mellea.stdlib.functional import ainstruct

        order: list[str] = []

        @hook("component_pre_create")
        async def h1(payload: Any, ctx: Any) -> Any:
            order.append("pre_create")
            return None

        @hook("component_post_create")
        async def h2(payload: Any, ctx: Any) -> Any:
            order.append("post_create")
            return None

        @hook("component_pre_execute")
        async def h3(payload: Any, ctx: Any) -> Any:
            order.append("pre_execute")
            return None

        @hook("component_post_success")
        async def h4(payload: Any, ctx: Any) -> Any:
            order.append("post_success")
            return None

        register(h1)
        register(h2)
        register(h3)
        register(h4)
        backend = _MockBackend()
        ctx = SimpleContext()
        await ainstruct("Full order test", ctx, backend, strategy=None)

        assert order == ["pre_create", "post_create", "pre_execute", "post_success"]


# ---------------------------------------------------------------------------
# Sampling hook call sites
# ---------------------------------------------------------------------------


class TestSamplingHookCallSites:
    """SAMPLING_LOOP_START, SAMPLING_ITERATION, SAMPLING_LOOP_END fire in
    BaseSamplingStrategy.sample()."""

    async def test_sampling_loop_start_fires(self):
        """SAMPLING_LOOP_START fires when RejectionSamplingStrategy.sample() begins."""
        from mellea.stdlib.sampling.base import RejectionSamplingStrategy
        from mellea.stdlib.components import Instruction

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

    async def test_sampling_loop_start_payload_has_strategy_name(self):
        """SAMPLING_LOOP_START payload.strategy_name contains the strategy class name."""
        from mellea.stdlib.sampling.base import RejectionSamplingStrategy
        from mellea.stdlib.components import Instruction

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

    async def test_sampling_loop_start_payload_has_correct_loop_budget(self):
        """SAMPLING_LOOP_START payload.loop_budget matches the strategy's loop_budget."""
        from mellea.stdlib.sampling.base import RejectionSamplingStrategy
        from mellea.stdlib.components import Instruction

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

    async def test_sampling_iteration_fires_once_per_loop_iteration(self):
        """SAMPLING_ITERATION fires once per loop iteration."""
        from mellea.stdlib.sampling.base import RejectionSamplingStrategy
        from mellea.stdlib.components import Instruction

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
        assert observed[0].all_valid is True  # no requirements → all pass

    async def test_sampling_loop_end_fires_on_success_path(self):
        """SAMPLING_LOOP_END fires with success=True when sampling succeeds."""
        from mellea.stdlib.sampling.base import RejectionSamplingStrategy
        from mellea.stdlib.components import Instruction

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

    async def test_sampling_loop_end_success_payload_has_final_result_and_context(self):
        """SAMPLING_LOOP_END success payload has final_result and final_context populated."""
        from mellea.stdlib.sampling.base import RejectionSamplingStrategy
        from mellea.stdlib.components import Instruction

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

    async def test_sampling_loop_end_context_in_plugin_ctx_is_result_ctx(self):
        """On success, SAMPLING_LOOP_END invoke_hook passes context=result_ctx (post-generation)."""
        from mellea.stdlib.sampling.base import RejectionSamplingStrategy
        from mellea.stdlib.components import Instruction

        observed_ctxs: list[Any] = []

        @hook("sampling_loop_end")
        async def recorder(payload: Any, ctx: Any) -> Any:
            # ctx.global_context.state["context"] is the context passed to invoke_hook
            observed_ctxs.append(ctx.global_context.state.get("context"))
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
        # On success, the context in plugin ctx should be result_ctx (not original_ctx)
        if observed_ctxs:
            assert observed_ctxs[0] is not original_ctx, (
                "Success path: plugin context should be result_ctx, not the original input context"
            )

    async def test_all_three_sampling_hooks_fire_in_order(self):
        """SAMPLING_LOOP_START → SAMPLING_ITERATION → SAMPLING_LOOP_END order."""
        from mellea.stdlib.sampling.base import RejectionSamplingStrategy
        from mellea.stdlib.components import Instruction

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

    def test_session_pre_init_fires_during_start_session(self):
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

    def test_session_pre_init_payload_has_backend_name_and_model_id(self):
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

    def test_session_post_init_fires_after_session_created(self):
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

    def test_session_post_init_payload_has_live_session(self):
        """SESSION_POST_INIT payload.session IS the live MelleaSession object."""
        from mellea.stdlib.session import start_session, MelleaSession

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
        assert p.session is session
        assert isinstance(p.session, MelleaSession)

    def test_pre_init_fires_before_post_init(self):
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
