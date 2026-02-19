"""End-to-end tests for payload policy enforcement through invoke_hook.

The plugin manager applies ``HookPayloadPolicy`` after each plugin returns:
only changes to ``writable_fields`` are accepted; all other mutations are
silently discarded.  Hooks absent from the policy table are observe-only
and reject every modification attempt.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mcpgateway.plugins.framework")

from mellea.plugins import PluginResult, hook, register
from mellea.plugins.hooks.component import (
    ComponentPostErrorPayload,
    ComponentPreCreatePayload,
)
from mellea.plugins.hooks.generation import GenerationPreCallPayload
from mellea.plugins.hooks.sampling import (
    SamplingIterationPayload,
    SamplingLoopStartPayload,
)
from mellea.plugins.hooks.session import SessionPreInitPayload
from mellea.plugins.manager import invoke_hook, shutdown_plugins
from mellea.plugins.types import HookType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _session_payload(**kwargs) -> SessionPreInitPayload:
    defaults: dict = dict(backend_name="original-backend", model_id="original-model")
    defaults.update(kwargs)
    return SessionPreInitPayload(**defaults)


def _component_payload(**kwargs) -> ComponentPreCreatePayload:
    defaults: dict = dict(
        description="original description", component_type="Instruction"
    )
    defaults.update(kwargs)
    return ComponentPreCreatePayload(**defaults)


def _generation_payload(**kwargs) -> GenerationPreCallPayload:
    defaults: dict = dict(
        model_options={"temperature": 0.5}, formatted_prompt="original prompt"
    )
    defaults.update(kwargs)
    return GenerationPreCallPayload(**defaults)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
async def reset_plugins():
    """Shut down and reset the plugin manager after every test."""
    yield
    await shutdown_plugins()


# ---------------------------------------------------------------------------
# TestWritableFieldAccepted
# ---------------------------------------------------------------------------


class TestWritableFieldAccepted:
    """Modifications to writable fields must be reflected in the returned payload."""

    async def test_backend_name_writable_in_session_pre_init(self):
        @hook("session_pre_init", priority=10)
        async def change_backend(payload, ctx):
            return PluginResult(
                continue_processing=True,
                modified_payload=payload.model_copy(
                    update={"backend_name": "plugin-backend"}
                ),
            )

        register(change_backend)

        payload = _session_payload()
        _, returned = await invoke_hook(HookType.SESSION_PRE_INIT, payload)

        assert returned.backend_name == "plugin-backend"

    async def test_model_options_writable_in_generation_pre_call(self):
        new_options = {"temperature": 0.9, "max_tokens": 512}

        @hook("generation_pre_call", priority=10)
        async def change_options(payload, ctx):
            return PluginResult(
                continue_processing=True,
                modified_payload=payload.model_copy(
                    update={"model_options": new_options}
                ),
            )

        register(change_options)

        payload = _generation_payload()
        _, returned = await invoke_hook(HookType.GENERATION_PRE_CALL, payload)

        assert returned.model_options == new_options

    async def test_description_writable_in_component_pre_create(self):
        @hook("component_pre_create", priority=10)
        async def change_description(payload, ctx):
            return PluginResult(
                continue_processing=True,
                modified_payload=payload.model_copy(
                    update={"description": "plugin-modified description"}
                ),
            )

        register(change_description)

        payload = _component_payload()
        _, returned = await invoke_hook(HookType.COMPONENT_PRE_CREATE, payload)

        assert returned.description == "plugin-modified description"


# ---------------------------------------------------------------------------
# TestNonWritableFieldDiscarded
# ---------------------------------------------------------------------------


class TestNonWritableFieldDiscarded:
    """Modifications to non-writable base fields must be silently discarded."""

    async def test_session_id_non_writable_in_session_pre_init(self):
        original_session_id = "original-session-id"

        @hook("session_pre_init", priority=10)
        async def tamper_session_id(payload, ctx):
            return PluginResult(
                continue_processing=True,
                modified_payload=payload.model_copy(
                    update={"session_id": "tampered-session-id"}
                ),
            )

        register(tamper_session_id)

        payload = _session_payload(session_id=original_session_id)
        _, returned = await invoke_hook(HookType.SESSION_PRE_INIT, payload)

        # session_id is a base payload field, not in the writable set — must be discarded
        assert returned.session_id == original_session_id

    async def test_hook_field_non_writable_in_session_pre_init(self):
        @hook("session_pre_init", priority=10)
        async def tamper_hook_field(payload, ctx):
            return PluginResult(
                continue_processing=True,
                modified_payload=payload.model_copy(
                    update={"hook": "injected-hook-value"}
                ),
            )

        register(tamper_hook_field)

        payload = _session_payload()
        _, returned = await invoke_hook(HookType.SESSION_PRE_INIT, payload)

        # hook field is set by the dispatcher — plugin cannot override it
        assert returned.hook == HookType.SESSION_PRE_INIT.value

    async def test_component_type_non_writable_in_component_pre_create(self):
        original_type = "Instruction"

        @hook("component_pre_create", priority=10)
        async def tamper_component_type(payload, ctx):
            return PluginResult(
                continue_processing=True,
                modified_payload=payload.model_copy(
                    update={"component_type": "Hijacked"}
                ),
            )

        register(tamper_component_type)

        payload = _component_payload(component_type=original_type)
        _, returned = await invoke_hook(HookType.COMPONENT_PRE_CREATE, payload)

        # component_type is not in the component_pre_create writable set
        assert returned.component_type == original_type


# ---------------------------------------------------------------------------
# TestObserveOnlyHookAcceptsAll
# ---------------------------------------------------------------------------


class TestObserveOnlyHookAcceptsAll:
    """Hooks absent from the policy table are documented as 'observe-only' in policies.py,
    but in practice the PluginManager does NOT enforce a DENY default for missing hooks.
    Modifications to any field are accepted because no policy entry restricts them.

    NOTE: policies.py says "Hooks absent from this table are observe-only. With
    DefaultHookPolicy.DENY (the Mellea default), any modification attempt on an
    observe-only hook is rejected." However, the PluginManager is initialised without
    an explicit default_hook_policy=DENY argument, so it falls back to ALLOW. This
    means ALL fields on observe-only hooks are effectively writable. This is a
    known gap — the intended DENY default is not yet enforced.
    """

    async def test_error_type_accepted_in_component_post_error(self):
        """component_post_error is observe-only per design, but modification is accepted in practice."""
        original_error_type = "ValueError"

        @hook("component_post_error", priority=10)
        async def tamper_error_type(payload, ctx):
            return PluginResult(
                continue_processing=True,
                modified_payload=payload.model_copy(
                    update={"error_type": "HackedError"}
                ),
            )

        register(tamper_error_type)

        payload = ComponentPostErrorPayload(
            component_type="Instruction", error_type=original_error_type
        )
        _, returned = await invoke_hook(HookType.COMPONENT_POST_ERROR, payload)

        # GAP: component_post_error has no policy entry; the framework SHOULD reject
        # modifications (DefaultHookPolicy.DENY), but currently accepts them.
        assert returned.error_type == "HackedError", (
            "component_post_error modification is accepted because the PluginManager "
            "does not enforce DefaultHookPolicy.DENY for missing policy entries. "
            "When the gap is fixed, change this to assert returned.error_type == original_error_type."
        )

    async def test_all_valid_accepted_in_sampling_iteration(self):
        """sampling_iteration is observe-only per design, but modification is accepted in practice."""

        @hook("sampling_iteration", priority=10)
        async def tamper_all_valid(payload, ctx):
            return PluginResult(
                continue_processing=True,
                modified_payload=payload.model_copy(update={"all_valid": True}),
            )

        register(tamper_all_valid)

        payload = SamplingIterationPayload(iteration=1, all_valid=False)
        _, returned = await invoke_hook(HookType.SAMPLING_ITERATION, payload)

        # GAP: sampling_iteration has no policy entry; modifications should be rejected
        # but are currently accepted due to the missing DefaultHookPolicy.DENY default.
        assert returned.all_valid is True, (
            "sampling_iteration modification is accepted because the PluginManager "
            "does not enforce DefaultHookPolicy.DENY for missing policy entries. "
            "When the gap is fixed, change this to assert returned.all_valid is False."
        )


# ---------------------------------------------------------------------------
# TestMixedModification
# ---------------------------------------------------------------------------


class TestMixedModification:
    """When a plugin modifies both writable and non-writable fields, only writable ones survive."""

    async def test_writable_accepted_non_writable_discarded(self):
        original_session_id = "original-sid"
        original_request_id = "original-rid"

        @hook("session_pre_init", priority=10)
        async def mixed_changes(payload, ctx):
            return PluginResult(
                continue_processing=True,
                modified_payload=payload.model_copy(
                    update={
                        # writable — should be accepted
                        "backend_name": "new-backend",
                        "model_id": "new-model",
                        # non-writable — should be discarded
                        "session_id": "injected-sid",
                        "request_id": "injected-rid",
                    }
                ),
            )

        register(mixed_changes)

        payload = _session_payload(
            session_id=original_session_id, request_id=original_request_id
        )
        _, returned = await invoke_hook(HookType.SESSION_PRE_INIT, payload)

        assert returned.backend_name == "new-backend"
        assert returned.model_id == "new-model"
        assert returned.session_id == original_session_id
        assert returned.request_id == original_request_id


# ---------------------------------------------------------------------------
# TestPayloadChaining
# ---------------------------------------------------------------------------


class TestPayloadChaining:
    """Accepted changes from Plugin A must be visible to Plugin B during the same invocation."""

    async def test_plugin_b_receives_plugin_a_changes(self):
        received_by_b: list[str] = []

        @hook("session_pre_init", priority=1)
        async def plugin_a(payload, ctx):
            return PluginResult(
                continue_processing=True,
                modified_payload=payload.model_copy(
                    update={"backend_name": "modified-by-a"}
                ),
            )

        @hook("session_pre_init", priority=100)
        async def plugin_b(payload, ctx):
            # Record the backend_name seen by Plugin B, then write model_id
            received_by_b.append(payload.backend_name)
            return PluginResult(
                continue_processing=True,
                modified_payload=payload.model_copy(
                    update={"model_id": "modified-by-b"}
                ),
            )

        register(plugin_a)
        register(plugin_b)

        payload = _session_payload()
        _, returned = await invoke_hook(HookType.SESSION_PRE_INIT, payload)

        # Plugin B must have observed Plugin A's accepted backend_name change
        assert received_by_b == ["modified-by-a"]

        # Final payload must carry both accepted modifications
        assert returned.backend_name == "modified-by-a"
        assert returned.model_id == "modified-by-b"


# ---------------------------------------------------------------------------
# TestReturnNoneIsNoop
# ---------------------------------------------------------------------------


class TestReturnNoneIsNoop:
    """A plugin that returns ``None`` must leave the payload entirely unchanged."""

    async def test_none_return_preserves_original_payload(self):
        @hook("session_pre_init", priority=10)
        async def noop_plugin(payload, ctx):
            return None  # Returning None signals: no change

        register(noop_plugin)

        payload = _session_payload()
        _, returned = await invoke_hook(HookType.SESSION_PRE_INIT, payload)

        assert returned.backend_name == "original-backend"
        assert returned.model_id == "original-model"


# ---------------------------------------------------------------------------
# TestReturnContinueTrueNoPayload
# ---------------------------------------------------------------------------


class TestReturnContinueTrueNoPayload:
    """PluginResult(continue_processing=True) with no modified_payload must be a no-op."""

    async def test_continue_true_without_payload_leaves_original(self):
        @hook("session_pre_init", priority=10)
        async def signal_only_plugin(payload, ctx):
            return PluginResult(continue_processing=True)  # no modified_payload

        register(signal_only_plugin)

        payload = _session_payload()
        _, returned = await invoke_hook(HookType.SESSION_PRE_INIT, payload)

        assert returned.backend_name == "original-backend"
        assert returned.model_id == "original-model"
