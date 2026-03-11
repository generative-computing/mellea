"""Tests for hook execution modes.

Execution order: SEQUENTIAL → TRANSFORM → AUDIT → CONCURRENT → FIRE_AND_FORGET

behavior summary
-----------------
- ``mode=SEQUENTIAL`` (default): serial, chained. Can block and modify payloads.

- ``mode=TRANSFORM``: serial, chained. Can modify payloads but cannot block — blocking
  results are suppressed.

- ``mode=AUDIT``: serial, observe-only. Cannot block or modify — violations are logged
  and payload modifications are discarded.

- ``mode=CONCURRENT``: parallel, fail-fast. Can block but cannot modify — payload
  modifications are discarded to avoid non-deterministic races.

- ``mode=FIRE_AND_FORGET``: background ``asyncio.create_task``. Cannot block or modify.
  Receives an isolated snapshot.
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
    async def test_blocking_plugin_raises_violation_error(self) -> None:
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
    async def test_non_blocking_plugin_returns_normally(self) -> None:
        """A hook that returns continue_processing=True does not raise."""
        invocations: list[str] = []

        @hook("session_pre_init", mode=PluginMode.SEQUENTIAL)
        async def observe_hook(payload, ctx):
            invocations.append("fired")
            return PluginResult(continue_processing=True)

        register(observe_hook)

        result, _returned_payload = await invoke_hook(
            HookType.SESSION_PRE_INIT, _session_payload()
        )

        assert invocations == ["fired"]
        assert result is not None
        assert result.continue_processing is True

    @pytest.mark.asyncio
    async def test_enforce_mode_writable_field_modification_is_accepted(self) -> None:
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
    async def test_enforce_stops_downstream_plugin_when_blocking(self) -> None:
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
    async def test_enforce_violation_error_carries_plugin_name(self) -> None:
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
    async def test_default_mode_is_sequential(self) -> None:
        """@hook without an explicit mode defaults to SEQUENTIAL and raises on violation."""

        @hook("session_pre_init")  # no mode= argument; default is SEQUENTIAL
        async def default_mode_blocker(payload, ctx):
            return block("Blocked by default-mode hook", code="DEFAULT_001")

        register(default_mode_blocker)

        with pytest.raises(PluginViolationError):
            await invoke_hook(HookType.SESSION_PRE_INIT, _session_payload())

    @pytest.mark.asyncio
    async def test_enforce_none_return_does_not_raise(self) -> None:
        """A hook returning None (no-op) in enforce mode does not raise."""

        @hook("session_pre_init", mode=PluginMode.SEQUENTIAL)
        async def silent_hook(payload, ctx):
            return None

        register(silent_hook)

        _result, returned_payload = await invoke_hook(
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
    async def test_blocking_permissive_plugin_does_not_raise(self) -> None:
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
    async def test_permissive_violation_does_not_stop_downstream_plugin(self) -> None:
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
    async def test_permissive_non_blocking_hook_fires_normally(self) -> None:
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
    async def test_multiple_permissive_blocking_plugins_all_fire(self) -> None:
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
    async def test_permissive_blocking_followed_by_enforce_observer(self) -> None:
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
    async def test_audit_modification_is_discarded(self) -> None:
        """AUDIT hooks are observe-only — payload modifications are discarded."""

        @hook("session_pre_init", mode=PluginMode.AUDIT)
        async def audit_modifier(payload, ctx):
            modified = payload.model_copy(update={"model_id": "audit-model"})
            return PluginResult(continue_processing=True, modified_payload=modified)

        register(audit_modifier)

        payload = _session_payload(model_id="original-model")
        _result, returned_payload = await invoke_hook(
            HookType.SESSION_PRE_INIT, payload
        )

        assert returned_payload.model_id == "original-model"


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
    async def test_fire_and_forget_hook_executes_as_background_task(self) -> None:
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
    async def test_fire_and_forget_blocking_does_not_raise(self) -> None:
        """A blocking fire-and-forget hook does NOT raise PluginViolationError.

        In OBSERVE mode violations are logged but never propagated — background
        tasks cannot halt the pipeline.
        """

        @hook("session_pre_init", mode=PluginMode.FIRE_AND_FORGET)
        async def faf_blocker(payload, ctx):
            return block("FAF block", code="FAF_001")

        register(faf_blocker)

        # Should complete without raising even though the hook returns block().
        result, _payload = await invoke_hook(
            HookType.SESSION_PRE_INIT, _session_payload()
        )
        assert result is not None and result.continue_processing is True

    @pytest.mark.asyncio
    async def test_fire_and_forget_writable_field_modification_is_not_applied(
        self,
    ) -> None:
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
    async def test_fire_and_forget_non_blocking_does_not_stop_downstream(self) -> None:
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
    async def test_fire_and_forget_mode_stored_correctly_in_hook_meta(self) -> None:
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
    async def test_fire_and_forget_none_return_is_noop(self) -> None:
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


# ---------------------------------------------------------------------------
# Transform mode
# ---------------------------------------------------------------------------


class TestTransformMode:
    """TRANSFORM plugins can modify payloads but cannot block the pipeline."""

    @pytest.mark.asyncio
    async def test_transform_modifies_writable_field(self) -> None:
        """A TRANSFORM hook can modify writable fields."""

        @hook("session_pre_init", mode=PluginMode.TRANSFORM)
        async def xform_modifier(payload, ctx):
            modified = payload.model_copy(update={"model_id": "transformed-model"})
            return PluginResult(continue_processing=True, modified_payload=modified)

        register(xform_modifier)

        payload = _session_payload(model_id="original-model")
        _result, returned_payload = await invoke_hook(
            HookType.SESSION_PRE_INIT, payload
        )

        assert returned_payload.model_id == "transformed-model"

    @pytest.mark.asyncio
    async def test_transform_cannot_block(self) -> None:
        """A TRANSFORM hook returning continue_processing=False is suppressed."""

        @hook("session_pre_init", mode=PluginMode.TRANSFORM)
        async def xform_blocker(payload, ctx):
            return block("Should be suppressed")

        register(xform_blocker)

        payload = _session_payload()
        # Should NOT raise — blocking is suppressed for TRANSFORM
        _result, returned_payload = await invoke_hook(
            HookType.SESSION_PRE_INIT, payload
        )
        assert returned_payload.backend_name == "test-backend"

    @pytest.mark.asyncio
    async def test_transform_chains_between_plugins(self) -> None:
        """Two TRANSFORM hooks chain their modifications."""

        @hook("session_pre_init", mode=PluginMode.TRANSFORM, priority=1)
        async def xform_a(payload, ctx):
            modified = payload.model_copy(update={"model_id": "step-a"})
            return PluginResult(continue_processing=True, modified_payload=modified)

        @hook("session_pre_init", mode=PluginMode.TRANSFORM, priority=2)
        async def xform_b(payload, ctx):
            # Should see step-a from the previous plugin
            modified = payload.model_copy(
                update={"model_options": {"seen": payload.model_id}}
            )
            return PluginResult(continue_processing=True, modified_payload=modified)

        register(xform_a)
        register(xform_b)

        payload = _session_payload()
        _result, returned_payload = await invoke_hook(
            HookType.SESSION_PRE_INIT, payload
        )

        assert returned_payload.model_id == "step-a"
        assert returned_payload.model_options == {"seen": "step-a"}

    @pytest.mark.asyncio
    async def test_transform_runs_after_sequential(self) -> None:
        """TRANSFORM hooks run after SEQUENTIAL hooks regardless of priority."""
        order: list[str] = []

        @hook("session_pre_init", mode=PluginMode.SEQUENTIAL, priority=99)
        async def seq_hook(payload, ctx):
            order.append("sequential")
            return None

        @hook("session_pre_init", mode=PluginMode.TRANSFORM, priority=1)
        async def xform_hook(payload, ctx):
            order.append("transform")
            return None

        register(seq_hook)
        register(xform_hook)

        await invoke_hook(HookType.SESSION_PRE_INIT, _session_payload())

        assert order == ["sequential", "transform"]

    @pytest.mark.asyncio
    async def test_transform_runs_before_audit(self) -> None:
        """TRANSFORM hooks run before AUDIT hooks."""
        order: list[str] = []

        @hook("session_pre_init", mode=PluginMode.AUDIT, priority=1)
        async def audit_hook(payload, ctx):
            order.append("audit")
            return None

        @hook("session_pre_init", mode=PluginMode.TRANSFORM, priority=99)
        async def xform_hook(payload, ctx):
            order.append("transform")
            return None

        register(audit_hook)
        register(xform_hook)

        await invoke_hook(HookType.SESSION_PRE_INIT, _session_payload())

        assert order == ["transform", "audit"]


# ---------------------------------------------------------------------------
# Concurrent mode
# ---------------------------------------------------------------------------


class TestConcurrentMode:
    """CONCURRENT plugins can block but cannot modify payloads."""

    @pytest.mark.asyncio
    async def test_concurrent_can_block(self) -> None:
        """A CONCURRENT hook returning continue_processing=False raises."""

        @hook("session_pre_init", mode=PluginMode.CONCURRENT)
        async def conc_blocker(payload, ctx):
            return block("Blocked by concurrent gate")

        register(conc_blocker)

        with pytest.raises(PluginViolationError):
            await invoke_hook(HookType.SESSION_PRE_INIT, _session_payload())

    @pytest.mark.asyncio
    async def test_concurrent_modification_is_discarded(self) -> None:
        """CONCURRENT hooks cannot modify payloads — modifications are discarded."""

        @hook("session_pre_init", mode=PluginMode.CONCURRENT)
        async def conc_modifier(payload, ctx):
            modified = payload.model_copy(update={"model_id": "concurrent-model"})
            return PluginResult(continue_processing=True, modified_payload=modified)

        register(conc_modifier)

        payload = _session_payload(model_id="original-model")
        _result, returned_payload = await invoke_hook(
            HookType.SESSION_PRE_INIT, payload
        )

        assert returned_payload.model_id == "original-model"

    @pytest.mark.asyncio
    async def test_concurrent_runs_after_audit(self) -> None:
        """CONCURRENT hooks run after AUDIT hooks."""
        order: list[str] = []

        @hook("session_pre_init", mode=PluginMode.CONCURRENT, priority=1)
        async def conc_hook(payload, ctx):
            order.append("concurrent")
            return None

        @hook("session_pre_init", mode=PluginMode.AUDIT, priority=99)
        async def audit_hook(payload, ctx):
            order.append("audit")
            return None

        register(conc_hook)
        register(audit_hook)

        await invoke_hook(HookType.SESSION_PRE_INIT, _session_payload())

        assert order == ["audit", "concurrent"]
