"""Unit tests for guardian adapter functions that require no model.

Covers: ``documents=`` kwarg on factuality_detection/factuality_correction,
and the policy_guardrails XOR validation logic.
"""

import pytest

from mellea.core.base import ModelOutputThunk
from mellea.stdlib.components import Document, Message
from mellea.stdlib.components.intrinsic import guardian
from mellea.stdlib.context import ChatContext


@pytest.fixture
def capture_intrinsic(monkeypatch):
    """Spy that replaces call_intrinsic and captures what it receives."""
    captured: dict = {}

    def fake_call_intrinsic(name, context, backend, /, kwargs=None, model_options=None):
        captured["name"] = name
        captured["context"] = context
        return {"score": "yes", "correction": "corrected"}

    monkeypatch.setattr(guardian, "call_intrinsic", fake_call_intrinsic)
    return captured


def _context_with_assistant():
    """Return a minimal manually-built context ending with an assistant message."""
    return (
        ChatContext()
        .add(Message("user", "Is Ozzy Osbourne alive?"))
        .add(Message("assistant", "Yes, he is."))
    )


# ---------------------------------------------------------------------------
# No-documents path: context is forwarded unchanged
# ---------------------------------------------------------------------------


def test_factuality_detection_no_documents_forwards_context(capture_intrinsic):
    ctx = _context_with_assistant()
    guardian.factuality_detection(ctx, object())
    assert capture_intrinsic["context"] is ctx


def test_factuality_correction_no_documents_forwards_context(capture_intrinsic):
    ctx = _context_with_assistant()
    guardian.factuality_correction(ctx, object())
    assert capture_intrinsic["context"] is ctx


# ---------------------------------------------------------------------------
# Correct intrinsic names are used
# ---------------------------------------------------------------------------


def test_factuality_detection_uses_correct_intrinsic_name(capture_intrinsic):
    guardian.factuality_detection(_context_with_assistant(), object())
    assert capture_intrinsic["name"] == "factuality-detection"


def test_factuality_correction_uses_correct_intrinsic_name(capture_intrinsic):
    guardian.factuality_correction(_context_with_assistant(), object())
    assert capture_intrinsic["name"] == "factuality-correction"


# ---------------------------------------------------------------------------
# documents= path: a new context is built with docs on the assistant message
# ---------------------------------------------------------------------------


def test_factuality_detection_with_docs_changes_context(capture_intrinsic):
    docs = [Document(text="Ozzy Osbourne passed away.", doc_id="1")]
    ctx = _context_with_assistant()
    guardian.factuality_detection(ctx, object(), documents=docs)

    passed = capture_intrinsic["context"]
    assert passed is not ctx


def test_factuality_correction_with_docs_changes_context(capture_intrinsic):
    ctx = _context_with_assistant()
    guardian.factuality_correction(ctx, object(), documents=["Ozzy Osbourne died."])

    passed = capture_intrinsic["context"]
    assert passed is not ctx


def test_factuality_detection_docs_attached_to_assistant_turn(capture_intrinsic):
    doc = Document(text="Ozzy Osbourne passed away.", doc_id="1")
    guardian.factuality_detection(_context_with_assistant(), object(), documents=[doc])

    passed = capture_intrinsic["context"]
    history = passed.view_for_generation() or []
    # The last element must be an assistant Message
    assert history, "Context should not be empty after document injection"
    last = history[-1]
    assert isinstance(last, Message)
    assert last.role == "assistant"
    assert last.content == "Yes, he is."
    # Document must be present in the message's parts
    doc_parts = [p for p in last.parts() if isinstance(p, Document)]
    assert len(doc_parts) == 1
    assert doc_parts[0].text == "Ozzy Osbourne passed away."


def test_factuality_correction_string_docs_coerced_to_document(capture_intrinsic):
    guardian.factuality_correction(
        _context_with_assistant(), object(), documents=["Plain string doc."]
    )

    passed = capture_intrinsic["context"]
    history = passed.view_for_generation() or []
    last = history[-1]
    doc_parts = [p for p in last.parts() if isinstance(p, Document)]
    assert len(doc_parts) == 1
    assert doc_parts[0].text == "Plain string doc."


def test_factuality_detection_preserves_preceding_messages(capture_intrinsic):
    """Injecting documents must not drop the user turn before the assistant response."""
    doc = Document(text="Reference.", doc_id="0")
    guardian.factuality_detection(_context_with_assistant(), object(), documents=[doc])

    passed = capture_intrinsic["context"]
    history = passed.view_for_generation() or []
    assert len(history) == 2, f"Expected 2 messages, got {len(history)}"
    assert isinstance(history[0], Message)
    assert history[0].role == "user"


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_inject_documents_raises_on_empty_context(capture_intrinsic):
    with pytest.raises(ValueError, match="Context is empty"):
        guardian.factuality_detection(ChatContext(), object(), documents=["doc"])


def test_inject_documents_raises_when_last_turn_is_not_assistant(capture_intrinsic):
    ctx = ChatContext().add(Message("user", "Hello"))
    with pytest.raises(ValueError, match="not an assistant response"):
        guardian.factuality_detection(ctx, object(), documents=["doc"])


def test_inject_documents_raises_on_uncomputed_thunk(capture_intrinsic):
    # value=None means not yet computed
    thunk = ModelOutputThunk(None)
    ctx = ChatContext().add(Message("user", "Is Ozzy Osbourne alive?")).add(thunk)
    with pytest.raises(ValueError, match="not been computed yet"):
        guardian.factuality_detection(ctx, object(), documents=["doc"])


# ---------------------------------------------------------------------------
# ModelOutputThunk path: session-generated context
# ---------------------------------------------------------------------------


def _context_with_thunk() -> ChatContext:
    """Return a context where the last turn is a computed ModelOutputThunk."""
    return (
        ChatContext()
        .add(Message("user", "Is Ozzy Osbourne alive?"))
        .add(ModelOutputThunk("Yes, he is."))
    )


def test_factuality_detection_thunk_changes_context(capture_intrinsic):
    ctx = _context_with_thunk()
    guardian.factuality_detection(ctx, object(), documents=["ref doc"])

    passed = capture_intrinsic["context"]
    assert passed is not ctx


def test_factuality_detection_thunk_preserves_user_turn(capture_intrinsic):
    guardian.factuality_detection(_context_with_thunk(), object(), documents=["ref"])

    history = capture_intrinsic["context"].view_for_generation() or []
    assert len(history) == 2
    assert isinstance(history[0], Message)
    assert history[0].role == "user"


def test_factuality_detection_thunk_attaches_doc_to_assistant_turn(capture_intrinsic):
    doc = Document(text="Ozzy Osbourne passed away.", doc_id="1")
    guardian.factuality_detection(_context_with_thunk(), object(), documents=[doc])

    history = capture_intrinsic["context"].view_for_generation() or []
    last = history[-1]
    assert isinstance(last, Message)
    assert last.role == "assistant"
    assert last.content == "Yes, he is."
    doc_parts = [p for p in last.parts() if isinstance(p, Document)]
    assert len(doc_parts) == 1
    assert doc_parts[0].text == "Ozzy Osbourne passed away."


# ---------------------------------------------------------------------------
# policy_guardrails: XOR validation error paths
# ---------------------------------------------------------------------------


@pytest.fixture
def capture_policy(monkeypatch):
    """Return a factory that makes call_intrinsic return a controlled result dict."""

    def _make(result: dict):
        monkeypatch.setattr(
            guardian,
            "call_intrinsic",
            lambda name, ctx, backend, /, kwargs=None, model_options=None: result,
        )

    return _make


def test_policy_guardrails_raises_when_both_label_and_score_present(capture_policy):
    capture_policy({"label": "Yes", "score": 0.9})
    ctx = ChatContext().add(Message("user", "Hello"))
    with pytest.raises(ValueError, match="found both"):
        guardian.policy_guardrails(ctx, object(), policy_text="no hate speech")


def test_policy_guardrails_raises_when_neither_label_nor_score_present(capture_policy):
    capture_policy({})
    ctx = ChatContext().add(Message("user", "Hello"))
    with pytest.raises(ValueError, match="found neither"):
        guardian.policy_guardrails(ctx, object(), policy_text="no hate speech")
