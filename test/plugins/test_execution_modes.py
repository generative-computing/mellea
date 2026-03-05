"""Tests for hook execution modes: enforce, permissive, and fire_and_forget.

behavior summary
-----------------
- ``mode=SEQUENTIAL`` (default): hook is awaited inline. If ``continue_processing=False``
  the ContextForge executor stops the chain and returns the blocking result; Mellea's
  ``invoke_hook`` then raises ``PluginViolationError``.

- ``mode=AUDIT``: hook is awaited inline. If ``continue_processing=False`` the
  ContextForge executor logs the violation but lets the loop continue.  The aggregate
  result returned to Mellea always has ``continue_processing=True``, so ``invoke_hook``
  does NOT raise.

- ``mode=FIRE_AND_FORGET``: mapped to ``PluginMode.OBSERVE`` at the ContextForge level.
  The hook is dispatched as a background ``asyncio.create_task`` and the pipeline
  continues immediately.  Violations are logged but never raised as exceptions, and
  payload modifications are discarded (the pipeline sees the original payload).
"""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("cpex.framework")

from mellea.plugins import PluginMode, PluginResult, block, hook, register
from mellea.plugins.base import PluginViolationError
from mellea.plugins.hooks.generation import GenerationPreCallPayload
from mellea.plugins.hooks.session import SessionPreInitPayload
from mellea.plugins.manager import invoke_hook, shutdown_plugins
from mellea.plugins.types import HookType, PluginMode

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _session_payload(**kwargs) -> SessionPreInitPayload:
    defaults: dict = dict(
        backend_name="test-backend", model_id="test-model", model_options=None
    )
    defaults.update(kwargs)
    return SessionPreInitPayload(**defaults)


def _generation_payload(**kwargs) -> GenerationPreCallPayload:
    defaults: dict = dict(model_options={"temperature": 0.7})
    defaults.update(kwargs)
    return GenerationPreCallPayload(**defaults)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
async def cleanup_plugins():
    """Reset plugin manager state after every test."""
    yield
    await shutdown_plugins()


# ---------------------------------------------------------------------------
# Enforce mode
# ---------------------------------------------------------------------------


class TestSequentialMode:
    """mode=SEQUENTIAL is the default. Violations raise PluginViolationError."""

    @pytest.mark.asyncio
    async def test_blocking_plugin_raises_violation_error(self):
        """A hook that returns block() in enforce mode causes invoke_hook to raise."""

        @hook("session_pre_init", mode=PluginMode.SEQUENTIAL)
        async def enforced_blocker(payload, ctx):
            return block("Access denied", code="AUTH_001")

        register(enforced_blocker)

        with pytest.raises(PluginViolationError) as exc_info:
            await invoke_hook(HookType.SESSION_PRE_INIT, _session_payload())

        err = exc_info.value
        assert err.hook_type == "session_pre_init"
        assert err.reason == "Access denied"
        assert err.code == "AUTH_001"

    @pytest.mark.asyncio
    async def test_non_blocking_plugin_returns_normally(self):
        """A hook that returns continue_processing=True does not raise."""
        invocations: list[str] = []

        @hook("session_pre_init", mode=PluginMode.SEQUENTIAL)
        async def observe_hook(payload, ctx):
            invocations.append("fired")
            return PluginResult(continue_processing=True)

        register(observe_hook)

        result, returned_payload = await invoke_hook(
            HookType.SESSION_PRE_INIT, _session_payload()
        )

        assert invocations == ["fired"]
        assert result is not None
        assert result.continue_processing is True

    @pytest.mark.asyncio
    async def test_enforce_mode_writable_field_modification_is_accepted(self):
        """A hook that modifies a writable field (model_id) has the change applied."""

        @hook("session_pre_init", mode=PluginMode.SEQUENTIAL)
        async def rewrite_model(payload, ctx):
            modified = payload.model_copy(update={"model_id": "gpt-4-turbo"})
            return PluginResult(continue_processing=True, modified_payload=modified)

        register(rewrite_model)

        payload = _session_payload(model_id="gpt-3.5")
        _result, returned_payload = await invoke_hook(
            HookType.SESSION_PRE_INIT, payload
        )

        # session_pre_init policy marks model_id as a writable field
        assert returned_payload.model_id == "gpt-4-turbo"

    @pytest.mark.asyncio
    async def test_enforce_stops_downstream_plugin_when_blocking(self):
        """When an enforce plugin blocks, downstream plugins do not fire."""
        downstream_calls: list[str] = []

        @hook("session_pre_init", mode=PluginMode.SEQUENTIAL)
        async def early_blocker(payload, ctx):
            return block("Stopped early", code="STOP_001")

        @hook("session_pre_init", mode=PluginMode.AUDIT)
        async def downstream_hook(payload, ctx):
            downstream_calls.append("fired")
            return None

        register(early_blocker)
        register(downstream_hook)

        with pytest.raises(PluginViolationError):
            await invoke_hook(HookType.SESSION_PRE_INIT, _session_payload())

        # Downstream hook must not have executed because SEQUENTIAL short-circuits the chain
        assert downstream_calls == []

    @pytest.mark.asyncio
    async def test_enforce_violation_error_carries_plugin_name(self):
        """PluginViolationError includes the plugin_name set by ContextForge."""

        @hook("generation_pre_call", mode=PluginMode.SEQUENTIAL)
        async def named_blocker(payload, ctx):
            return block("Rate limit exceeded", code="RATE_001")

        register(named_blocker)

        with pytest.raises(PluginViolationError) as exc_info:
            await invoke_hook(HookType.GENERATION_PRE_CALL, _generation_payload())

        # ContextForge sets violation.plugin_name from the registered handler's name
        assert exc_info.value.plugin_name != ""

    @pytest.mark.asyncio
    async def test_default_mode_is_sequential(self):
        """@hook without an explicit mode defaults to SEQUENTIAL and raises on violation."""

        @hook("session_pre_init")  # no mode= argument; default is SEQUENTIAL
        async def default_mode_blocker(payload, ctx):
            return block("Blocked by default-mode hook", code="DEFAULT_001")

        register(default_mode_blocker)

        with pytest.raises(PluginViolationError):
            await invoke_hook(HookType.SESSION_PRE_INIT, _session_payload())

    @pytest.mark.asyncio
    async def test_enforce_none_return_does_not_raise(self):
        """A hook returning None (no-op) in enforce mode does not raise."""

        @hook("session_pre_init", mode=PluginMode.SEQUENTIAL)
        async def silent_hook(payload, ctx):
            return None

        register(silent_hook)

        result, returned_payload = await invoke_hook(
            HookType.SESSION_PRE_INIT, _session_payload()
        )
        # No exception; payload is unchanged
        assert returned_payload.backend_name == "test-backend"


# ---------------------------------------------------------------------------
# Permissive mode
# ---------------------------------------------------------------------------


class TestAuditMode:
    """mode=AUDIT: violations are logged but do not raise or stop execution."""

    @pytest.mark.asyncio
    async def test_blocking_permissive_plugin_does_not_raise(self):
        """A blocking plugin in permissive mode must not raise PluginViolationError."""

        @hook("session_pre_init", mode=PluginMode.AUDIT)
        async def permissive_blocker(payload, ctx):
            return block("Would block, but permissive", code="PERM_001")

        register(permissive_blocker)

        # Must not raise
        result, _returned_payload = await invoke_hook(
            HookType.SESSION_PRE_INIT, _session_payload()
        )
        # ContextForge execute() loop always returns continue_processing=True after
        # a permissive violation because it continues iterating to end of the chain.
        assert result is not None
        assert result.continue_processing is True

    @pytest.mark.asyncio
    async def test_permissive_violation_does_not_stop_downstream_plugin(self):
        """Downstream plugins still fire after a permissive plugin signals a violation."""
        downstream_calls: list[str] = []

        @hook("session_pre_init", mode=PluginMode.AUDIT, priority=5)
        async def early_permissive_blocker(payload, ctx):
            return block("Soft block", code="SOFT_001")

        @hook("session_pre_init", mode=PluginMode.SEQUENTIAL, priority=10)
        async def downstream_hook(payload, ctx):
            downstream_calls.append("fired")
            return None

        register(early_permissive_blocker)
        register(downstream_hook)

        # Should not raise
        await invoke_hook(HookType.SESSION_PRE_INIT, _session_payload())

        # Downstream hook must have executed despite the earlier permissive block
        assert downstream_calls == ["fired"]

    @pytest.mark.asyncio
    async def test_permissive_non_blocking_hook_fires_normally(self):
        """A permissive hook that continues fires and the call succeeds."""
        invocations: list[str] = []

        @hook("session_pre_init", mode=PluginMode.AUDIT)
        async def permissive_observer(payload, ctx):
            invocations.append(payload.backend_name)
            return PluginResult(continue_processing=True)

        register(permissive_observer)

        await invoke_hook(HookType.SESSION_PRE_INIT, _session_payload())

        assert invocations == ["test-backend"]

    @pytest.mark.asyncio
    async def test_multiple_permissive_blocking_plugins_all_fire(self):
        """Multiple permissive blocking plugins all execute; no exception is raised."""
        fires: list[str] = []

        @hook("session_pre_init", mode=PluginMode.AUDIT, priority=5)
        async def first_permissive(payload, ctx):
            fires.append("first")
            return block("First block", code="P001")

        @hook("session_pre_init", mode=PluginMode.AUDIT, priority=10)
        async def second_permissive(payload, ctx):
            fires.append("second")
            return block("Second block", code="P002")

        register(first_permissive)
        register(second_permissive)

        await invoke_hook(HookType.SESSION_PRE_INIT, _session_payload())

        assert fires == ["first", "second"]

    @pytest.mark.asyncio
    async def test_permissive_blocking_followed_by_enforce_observer(self):
        """A permissive blocker followed by a non-blocking enforce hook: both fire, and enforce goes first."""
        order: list[str] = []

        @hook("session_pre_init", mode=PluginMode.AUDIT)
        async def permissive_block(payload, ctx):
            order.append("permissive")
            return block("Soft block", code="PERM_SIBLING")

        @hook("session_pre_init", mode=PluginMode.SEQUENTIAL)
        async def enforce_observer(payload, ctx):
            order.append("enforce")
            return PluginResult(continue_processing=True)

        register(permissive_block)
        register(enforce_observer)

        await invoke_hook(HookType.SESSION_PRE_INIT, _session_payload())

        assert order == ["enforce", "permissive"]

    @pytest.mark.asyncio
    async def test_permissive_continuing_hook_modifies_writable_field(self):
        """A permissive hook that does NOT block and modifies a writable field applies the change."""

        @hook("session_pre_init", mode=PluginMode.AUDIT)
        async def permissive_modifier(payload, ctx):
            modified = payload.model_copy(update={"model_id": "permissive-model"})
            return PluginResult(continue_processing=True, modified_payload=modified)

        register(permissive_modifier)

        payload = _session_payload(model_id="original-model")
        _result, returned_payload = await invoke_hook(
            HookType.SESSION_PRE_INIT, payload
        )

        assert returned_payload.model_id == "permissive-model"


# ---------------------------------------------------------------------------
# Fire-and-forget mode
# ---------------------------------------------------------------------------


class TestFireAndForgetMode:
    """mode=FIRE_AND_FORGET: mapped to PluginMode.OBSERVE at the ContextForge level.

    The Mellea registry maps PluginMode.FIRE_AND_FORGET to the ContextForge
    PluginMode.OBSERVE (see mellea/plugins/registry.py).  Consequently:

    - The hook is dispatched as a background asyncio.create_task (not awaited inline).
    - Violations are logged but never raised as PluginViolationError.
    - Payload modifications are discarded; the pipeline sees the original payload.
    """

    @pytest.mark.asyncio
    async def test_fire_and_forget_hook_executes_as_background_task(self):
        """A fire-and-forget hook fires as a background task and records its invocation."""
        invocations: list[str] = []

        @hook("session_pre_init", mode=PluginMode.FIRE_AND_FORGET)
        async def faf_observer(payload, ctx):
            invocations.append("fired")
            return PluginResult(continue_processing=True)

        register(faf_observer)

        await invoke_hook(HookType.SESSION_PRE_INIT, _session_payload())

        # The hook runs as a background asyncio task; yield to the event loop to
        # allow it to complete before asserting.
        await asyncio.sleep(0.05)
        assert invocations == ["fired"]

    @pytest.mark.asyncio
    async def test_fire_and_forget_blocking_does_not_raise(self):
        """A blocking fire-and-forget hook does NOT raise PluginViolationError.

        In OBSERVE mode violations are logged but never propagated — background
        tasks cannot halt the pipeline.
        """

        @hook("session_pre_init", mode=PluginMode.FIRE_AND_FORGET)
        async def faf_blocker(payload, ctx):
            return block("FAF block", code="FAF_001")

        register(faf_blocker)

        # Should complete without raising even though the hook returns block().
        result, payload = await invoke_hook(
            HookType.SESSION_PRE_INIT, _session_payload()
        )
        assert result.continue_processing is True

    @pytest.mark.asyncio
    async def test_fire_and_forget_writable_field_modification_is_not_applied(self):
        """A fire-and-forget hook that modifies a writable field does NOT affect the pipeline.

        In OBSERVE mode the hook receives a copy of the payload; its modifications are
        discarded and the original payload is returned to the caller unchanged.
        """

        @hook("session_pre_init", mode=PluginMode.FIRE_AND_FORGET)
        async def faf_modifier(payload, ctx):
            modified = payload.model_copy(update={"backend_name": "modified-backend"})
            return PluginResult(continue_processing=True, modified_payload=modified)

        register(faf_modifier)

        payload = _session_payload(backend_name="original-backend")
        _result, returned_payload = await invoke_hook(
            HookType.SESSION_PRE_INIT, payload
        )

        assert returned_payload.backend_name == "original-backend"

    @pytest.mark.asyncio
    async def test_fire_and_forget_non_blocking_does_not_stop_downstream(self):
        """A non-blocking fire-and-forget hook lets downstream plugins fire."""
        order: list[str] = []

        @hook("session_pre_init", mode=PluginMode.FIRE_AND_FORGET, priority=5)
        async def faf_first(payload, ctx):
            order.append("faf")
            return PluginResult(continue_processing=True)

        @hook("session_pre_init", mode=PluginMode.SEQUENTIAL, priority=10)
        async def enforce_second(payload, ctx):
            order.append("enforce")
            return PluginResult(continue_processing=True)

        register(faf_first)
        register(enforce_second)

        await invoke_hook(HookType.SESSION_PRE_INIT, _session_payload())

        await asyncio.sleep(0.05)
        assert order == ["enforce", "faf"]

    @pytest.mark.asyncio
    async def test_fire_and_forget_mode_stored_correctly_in_hook_meta(self):
        """HookMeta records PluginMode.FIRE_AND_FORGET on the decorated function.

        Verifies that the Mellea-layer decorator stores the correct mode enum value
        regardless of how the ContextForge adapter maps it at registration time.
        """
        from mellea.plugins.decorators import HookMeta

        @hook("session_pre_init", mode=PluginMode.FIRE_AND_FORGET, priority=25)
        async def faf_fn(payload, ctx):
            return None

        meta: HookMeta = faf_fn._mellea_hook_meta
        assert meta.mode == PluginMode.FIRE_AND_FORGET
        assert meta.hook_type == "session_pre_init"
        assert meta.priority == 25

    @pytest.mark.asyncio
    async def test_fire_and_forget_none_return_is_noop(self):
        """A fire-and-forget hook returning None leaves the payload unchanged."""

        @hook("session_pre_init", mode=PluginMode.FIRE_AND_FORGET)
        async def faf_noop(payload, ctx):
            return None

        register(faf_noop)

        payload = _session_payload(backend_name="unchanged")
        _result, returned_payload = await invoke_hook(
            HookType.SESSION_PRE_INIT, payload
        )

        assert returned_payload.backend_name == "unchanged"
