"""Tests for hook payload models."""

import weakref

import pytest
from pydantic import BeforeValidator, ValidationError

from mellea.plugins.base import MelleaBasePayload
from mellea.plugins.hooks.component import (
    ComponentPostErrorPayload,
    ComponentPostSuccessPayload,
    ComponentPreExecutePayload,
)
from mellea.plugins.hooks.generation import GenerationPreCallPayload
from mellea.plugins.hooks.sampling import (
    SamplingIterationPayload,
    SamplingLoopEndPayload,
    SamplingLoopStartPayload,
    SamplingRepairPayload,
)
from mellea.plugins.hooks.session import (
    SessionCleanupPayload,
    SessionPostInitPayload,
    SessionPreInitPayload,
    SessionResetPayload,
)
from mellea.plugins.hooks.validation import ValidationPreCheckPayload


class TestMelleaBasePayload:
    def test_frozen(self):
        payload = MelleaBasePayload(request_id="test-123")
        with pytest.raises(ValidationError):
            payload.request_id = "new-value"

    def test_defaults(self):
        payload = MelleaBasePayload()
        assert payload.session_id is None
        assert payload.request_id == ""
        assert payload.hook == ""
        assert payload.user_metadata == {}
        assert payload.timestamp is not None

    def test_model_copy(self):
        payload = MelleaBasePayload(request_id="test-123", hook="test_hook")
        modified = payload.model_copy(update={"request_id": "new-123"})
        assert modified.request_id == "new-123"
        assert modified.hook == "test_hook"
        # Original unchanged
        assert payload.request_id == "test-123"


class TestSessionPreInitPayload:
    def test_creation(self):
        payload = SessionPreInitPayload(
            backend_name="openai", model_id="gpt-4", model_options={"temperature": 0.7}
        )
        assert payload.backend_name == "openai"
        assert payload.model_id == "gpt-4"
        assert payload.model_options == {"temperature": 0.7}

    def test_frozen(self):
        payload = SessionPreInitPayload(backend_name="openai", model_id="gpt-4")
        with pytest.raises(ValidationError):
            payload.backend_name = "hf"

    def test_model_copy_writable_fields(self):
        payload = SessionPreInitPayload(
            backend_name="openai", model_id="gpt-4", model_options=None
        )
        modified = payload.model_copy(
            update={"model_id": "gpt-3.5", "model_options": {"temperature": 0.5}}
        )
        assert modified.model_id == "gpt-3.5"
        assert modified.model_options == {"temperature": 0.5}


class TestGenerationPreCallPayload:
    def test_creation(self):
        payload = GenerationPreCallPayload(
            model_options={"max_tokens": 100}, format=None
        )
        assert payload.model_options == {"max_tokens": 100}
        assert payload.format is None


# ---------------------------------------------------------------------------
# Helper for WeakProxy field detection
# ---------------------------------------------------------------------------


def _is_weak_proxy_field(payload_cls, field_name: str) -> bool:
    """Return True if *field_name* on *payload_cls* uses the WeakProxy validator."""
    field_info = payload_cls.model_fields.get(field_name)
    if field_info is None:
        return False
    return any(
        isinstance(m, BeforeValidator)
        and getattr(m.func, "__name__", None) == "_to_weak_proxy"
        for m in field_info.metadata
    )


class _Dummy:
    """A simple weakref-able object for testing WeakProxy payload fields."""

    def __init__(self, name: str = "dummy"):
        self.name = name

    def __str__(self) -> str:
        return self.name


# ---------------------------------------------------------------------------
# WeakProxy payload field tests
# ---------------------------------------------------------------------------

# Map of (payload_class, {field_name: ...}) for every WeakProxy field.
_WEAK_PROXY_PAYLOADS: list[tuple[type, list[str]]] = [
    (GenerationPreCallPayload, ["action", "context"]),
    (ComponentPreExecutePayload, ["action", "context", "strategy"]),
    (ComponentPostSuccessPayload, ["action", "context_before", "context_after"]),
    (ComponentPostErrorPayload, ["action", "context"]),
    (SessionPostInitPayload, ["session"]),
    (SessionResetPayload, ["previous_context"]),
    (SessionCleanupPayload, ["context"]),
    (SamplingLoopStartPayload, ["action", "context"]),
    (SamplingIterationPayload, ["action"]),
    (SamplingRepairPayload, ["failed_action", "repair_action", "repair_context"]),
    (SamplingLoopEndPayload, ["final_action", "final_context"]),
    (ValidationPreCheckPayload, ["target", "context"]),
]


class TestWeakProxyFields:
    """Tests for WeakProxy fields across all payload types."""

    @pytest.mark.parametrize(
        "payload_cls, fields",
        _WEAK_PROXY_PAYLOADS,
        ids=[cls.__name__ for cls, _ in _WEAK_PROXY_PAYLOADS],
    )
    def test_field_has_weak_proxy_validator(self, payload_cls, fields):
        """Every listed field must use the _to_weak_proxy BeforeValidator."""
        for field_name in fields:
            assert _is_weak_proxy_field(payload_cls, field_name), (
                f"{payload_cls.__name__}.{field_name} is not a WeakProxy field"
            )

    @pytest.mark.parametrize(
        "payload_cls, fields",
        _WEAK_PROXY_PAYLOADS,
        ids=[cls.__name__ for cls, _ in _WEAK_PROXY_PAYLOADS],
    )
    def test_stores_weakref_proxy(self, payload_cls, fields):
        """WeakProxy fields should store a weakref.proxy of the input object."""
        dummy = _Dummy()
        kwargs = dict.fromkeys(fields, dummy)
        payload = payload_cls(**kwargs)
        for field_name in fields:
            value = getattr(payload, field_name)
            assert isinstance(value, weakref.ProxyType), (
                f"{payload_cls.__name__}.{field_name} should be a weakref.proxy, "
                f"got {type(value).__name__}"
            )

    @pytest.mark.parametrize(
        "payload_cls, fields",
        _WEAK_PROXY_PAYLOADS,
        ids=[cls.__name__ for cls, _ in _WEAK_PROXY_PAYLOADS],
    )
    def test_isinstance_works_through_proxy(self, payload_cls, fields):
        """isinstance() should see through the proxy to the underlying type."""
        dummy = _Dummy("test-obj")
        kwargs = dict.fromkeys(fields, dummy)
        payload = payload_cls(**kwargs)
        for field_name in fields:
            value = getattr(payload, field_name)
            assert isinstance(value, _Dummy), (
                f"isinstance check failed for {payload_cls.__name__}.{field_name}"
            )

    @pytest.mark.parametrize(
        "payload_cls, fields",
        _WEAK_PROXY_PAYLOADS,
        ids=[cls.__name__ for cls, _ in _WEAK_PROXY_PAYLOADS],
    )
    def test_type_name_returns_proxy_type(self, payload_cls, fields):
        """type(proxy).__name__ returns 'ProxyType', not the underlying class.

        This documents the known limitation — framework code must capture
        type names before values are wrapped in WeakProxy.
        """
        dummy = _Dummy()
        kwargs = dict.fromkeys(fields, dummy)
        payload = payload_cls(**kwargs)
        for field_name in fields:
            value = getattr(payload, field_name)
            assert type(value).__name__ == "ProxyType"

    @pytest.mark.parametrize(
        "payload_cls, fields",
        _WEAK_PROXY_PAYLOADS,
        ids=[cls.__name__ for cls, _ in _WEAK_PROXY_PAYLOADS],
    )
    def test_attribute_access_through_proxy(self, payload_cls, fields):
        """Attribute access should forward through the proxy."""
        dummy = _Dummy("forwarded")
        kwargs = dict.fromkeys(fields, dummy)
        payload = payload_cls(**kwargs)
        for field_name in fields:
            value = getattr(payload, field_name)
            assert value.name == "forwarded"

    @pytest.mark.parametrize(
        "payload_cls, fields",
        _WEAK_PROXY_PAYLOADS,
        ids=[cls.__name__ for cls, _ in _WEAK_PROXY_PAYLOADS],
    )
    def test_none_not_wrapped(self, payload_cls, fields):
        """None values should remain None, not a proxy to None."""
        # All WeakProxy fields default to None
        payload = payload_cls()
        for field_name in fields:
            value = getattr(payload, field_name)
            assert value is None, (
                f"{payload_cls.__name__}.{field_name} default should be None"
            )

    @pytest.mark.parametrize(
        "payload_cls, fields",
        _WEAK_PROXY_PAYLOADS,
        ids=[cls.__name__ for cls, _ in _WEAK_PROXY_PAYLOADS],
    )
    def test_double_wrapping_is_safe(self, payload_cls, fields):
        """Passing a weakref.proxy as input should not crash (double-wrap)."""
        dummy = _Dummy()
        proxy = weakref.proxy(dummy)
        kwargs = dict.fromkeys(fields, proxy)
        # Should not raise — _to_weak_proxy catches TypeError and returns as-is
        payload = payload_cls(**kwargs)
        for field_name in fields:
            value = getattr(payload, field_name)
            assert value.name == "dummy"

    def test_gc_raises_reference_error(self):
        """Accessing a WeakProxy field after the referent is GC'd raises ReferenceError.

        Uses a standalone weakref.proxy (not embedded in a payload) to avoid
        Pydantic's internal model storage keeping a strong reference.
        """
        import gc

        dummy = _Dummy("will-be-collected")
        proxy = weakref.proxy(dummy)
        assert proxy.name == "will-be-collected"
        del dummy
        gc.collect()
        with pytest.raises(ReferenceError):
            _ = proxy.name


class TestWeakProxyIsolation:
    """cpex wrap_payload_for_isolation must handle WeakProxy fields."""

    def test_isolation_does_not_deepcopy_proxy(self):
        """wrap_payload_for_isolation should pass WeakProxy values through."""
        cpex_memory = pytest.importorskip("cpex.framework.memory")
        dummy = _Dummy("isolation-test")
        payload = GenerationPreCallPayload(action=dummy, context=dummy)
        # Should not raise (previously crashed with RLock deepcopy error)
        isolated = cpex_memory.wrap_payload_for_isolation(payload)
        assert isolated.action.name == "isolation-test"
        assert isolated.context.name == "isolation-test"


class TestWeakProxyWritableReadBack:
    """Writable WeakProxy fields should be safely readable after hook return."""

    def test_modified_proxy_field_readable(self):
        """A plugin-modified WeakProxy field should be readable after model_copy.

        Note: ``model_copy(update=...)`` bypasses Pydantic validators, so the
        replacement value is stored as-is (not wrapped in a proxy).  This is
        correct — the framework reads back the concrete replacement object.
        """
        original = _Dummy("original")
        replacement = _Dummy("replaced")
        payload = ComponentPreExecutePayload(action=original)
        modified = payload.model_copy(update={"action": replacement})
        assert modified.action.name == "replaced"
        # Original payload still holds the proxy to the original object
        assert isinstance(payload.action, weakref.ProxyType)
        assert payload.action.name == "original"

    def test_none_fallback_on_gc(self):
        """If the proxy referent is GC'd, the field becomes unusable (None guard)."""
        dummy = _Dummy("ephemeral")
        payload = ComponentPreExecutePayload(action=dummy)
        del dummy
        # The proxy is now dangling — reading .action won't be None (it's a
        # dead proxy), but accessing attributes raises ReferenceError.
        # Framework code guards with `if value is not None` which returns True
        # for dead proxies, but attribute access raises.
        with pytest.raises(ReferenceError):
            _ = payload.action.name

    def test_writable_weak_proxy_fields_in_policies(self):
        """Writable WeakProxy fields should be listed in their hook policies."""
        from mellea.plugins.policies import MELLEA_HOOK_PAYLOAD_POLICIES

        policy = MELLEA_HOOK_PAYLOAD_POLICIES["component_pre_execute"]
        assert "action" in policy.writable_fields
        assert "context" in policy.writable_fields
        assert "strategy" in policy.writable_fields

        policy = MELLEA_HOOK_PAYLOAD_POLICIES["sampling_repair"]
        assert "repair_action" in policy.writable_fields
        assert "repair_context" in policy.writable_fields
