"""End-to-end tests for payload policy enforcement through invoke_hook.

The plugin manager applies ``HookPayloadPolicy`` after each plugin returns:
only changes to ``writable_fields`` are accepted; all other mutations are
silently discarded.  Hooks absent from the policy table are observe-only
and reject every modification attempt.
"""

from __future__ import annotations

import pytest

pytest.importorskip("cpex.framework")

from mellea.plugins import PluginMode, PluginResult, hook, register
from mellea.plugins.hooks.component import ComponentPostErrorPayload
from mellea.plugins.hooks.generation import GenerationPreCallPayload
from mellea.plugins.hooks.sampling import SamplingIterationPayload
from mellea.plugins.hooks.session import SessionPreInitPayload
from mellea.plugins.manager import invoke_hook
from mellea.plugins.types import HookType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _session_payload(**kwargs) -> SessionPreInitPayload:
    defaults: dict = dict(backend_name="original-backend", model_id="original-model")
    defaults.update(kwargs)
    return SessionPreInitPayload(**defaults)


def _generation_payload(**kwargs) -> GenerationPreCallPayload:
    defaults: dict = dict(
        model_options={"temperature": 0.5}, formatted_prompt="original prompt"
    )
    defaults.update(kwargs)
    return GenerationPreCallPayload(**defaults)


# ---------------------------------------------------------------------------
# TestWritableFieldAccepted
# ---------------------------------------------------------------------------


class TestWritableFieldAccepted:
    """Modifications to writable fields must be reflected in the returned payload."""

    async def test_model_id_writable_in_session_pre_init(self) -> None:
        @hook("session_pre_init", priority=10)
        async def change_model(payload, ctx):
            return PluginResult(
                continue_processing=True,
                modified_payload=payload.model_copy(
                    update={"model_id": "plugin-model"}
                ),
            )

        register(change_model)

        payload = _session_payload()
        _, returned = await invoke_hook(HookType.SESSION_PRE_INIT, payload)

        assert returned.model_id == "plugin-model"

    async def test_model_options_writable_in_generation_pre_call(self) -> None:
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


# ---------------------------------------------------------------------------
# TestNonWritableFieldDiscarded
# ---------------------------------------------------------------------------


class TestNonWritableFieldDiscarded:
    """Modifications to non-writable base fields must be silently discarded."""

    async def test_session_id_non_writable_in_session_pre_init(self) -> None:
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

    async def test_hook_field_non_writable_in_session_pre_init(self) -> None:
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


# ---------------------------------------------------------------------------
# TestObserveOnlyHookAcceptsAll
# ---------------------------------------------------------------------------


class TestObserveOnlyHookRejectsModifications:
    """Hooks absent from the policy table are 'observe-only': the PluginManager enforces
    DefaultHookPolicy.DENY, silently discarding all plugin-proposed modifications.
    """

    async def test_error_type_rejected_in_component_post_error(self) -> None:
        """component_post_error is observe-only — plugin modifications are discarded."""
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

        # DefaultHookPolicy.DENY is now enforced: modification is silently discarded.
        assert returned.error_type == original_error_type

    async def test_all_validations_passed_rejected_in_sampling_iteration(self) -> None:
        """sampling_iteration is observe-only — plugin modifications are discarded."""

        @hook("sampling_iteration", priority=10)
        async def tamper_all_validations_passed(payload, ctx):
            return PluginResult(
                continue_processing=True,
                modified_payload=payload.model_copy(
                    update={"all_validations_passed": True}
                ),
            )

        register(tamper_all_validations_passed)

        payload = SamplingIterationPayload(iteration=1, all_validations_passed=False)
        _, returned = await invoke_hook(HookType.SAMPLING_ITERATION, payload)

        # DefaultHookPolicy.DENY is now enforced: modification is silently discarded.
        assert returned.all_validations_passed is False


# ---------------------------------------------------------------------------
# TestMixedModification
# ---------------------------------------------------------------------------


class TestMixedModification:
    """When a plugin modifies both writable and non-writable fields, only writable ones survive."""

    async def test_writable_accepted_non_writable_discarded(self) -> None:
        original_session_id = "original-sid"
        original_request_id = "original-rid"

        @hook("session_pre_init", priority=10)
        async def mixed_changes(payload, ctx):
            return PluginResult(
                continue_processing=True,
                modified_payload=payload.model_copy(
                    update={
                        # writable — should be accepted
                        "model_id": "new-model",
                        "model_options": {"temperature": 0.9},
                        # non-writable — should be discarded
                        "backend_name": "new-backend",
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

        assert returned.model_id == "new-model"
        assert returned.model_options == {"temperature": 0.9}
        assert returned.backend_name == "original-backend"
        assert returned.session_id == original_session_id
        assert returned.request_id == original_request_id


# ---------------------------------------------------------------------------
# TestPayloadChaining
# ---------------------------------------------------------------------------


class TestPayloadChaining:
    """Accepted changes from Plugin A must be visible to Plugin B during the same invocation."""

    async def test_plugin_b_receives_plugin_a_changes(self) -> None:
        received_by_b: list[str] = []

        @hook("session_pre_init", mode=PluginMode.TRANSFORM, priority=1)
        async def plugin_a(payload, ctx):
            return PluginResult(
                continue_processing=True,
                modified_payload=payload.model_copy(
                    update={"model_id": "modified-by-a"}
                ),
            )

        @hook("session_pre_init", mode=PluginMode.TRANSFORM, priority=100)
        async def plugin_b(payload, ctx):
            # Record the model_id seen by Plugin B, then write model_options
            received_by_b.append(payload.model_id)
            return PluginResult(
                continue_processing=True,
                modified_payload=payload.model_copy(
                    update={"model_options": {"temperature": 0.9}}
                ),
            )

        register(plugin_a)
        register(plugin_b)

        payload = _session_payload()
        _, returned = await invoke_hook(HookType.SESSION_PRE_INIT, payload)

        # Plugin B must have observed Plugin A's accepted model_id change
        assert received_by_b == ["modified-by-a"]

        # Final payload must carry both accepted modifications
        assert returned.model_id == "modified-by-a"
        assert returned.model_options == {"temperature": 0.9}


# ---------------------------------------------------------------------------
# TestReturnNoneIsNoop
# ---------------------------------------------------------------------------


class TestReturnNoneIsNoop:
    """A plugin that returns ``None`` must leave the payload entirely unchanged."""

    async def test_none_return_preserves_original_payload(self) -> None:
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

    async def test_continue_true_without_payload_leaves_original(self) -> None:
        @hook("session_pre_init", priority=10)
        async def signal_only_plugin(payload, ctx):
            return PluginResult(continue_processing=True)  # no modified_payload

        register(signal_only_plugin)

        payload = _session_payload()
        _, returned = await invoke_hook(HookType.SESSION_PRE_INIT, payload)

        assert returned.backend_name == "original-backend"
        assert returned.model_id == "original-model"
