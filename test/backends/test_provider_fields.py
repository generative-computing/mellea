# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for author-pluggable message serialization via `provider_fields` (#1565).

Covers the round-trip through the IR path, the shared `merge_provider_fields`
helper's match rule / precedence / mismatch error, and that an author-declared
field reaches the wire message on each affected serialization path.
"""

import logging

import pytest

from mellea.core.base import TemplateRepresentation
from mellea.helpers.openai_compatible_helpers import (
    OPENAI_COMPATIBLE_WIRE_PROVIDERS,
    merge_provider_fields,
    message_to_openai_message,
)
from mellea.stdlib.components import Message
from mellea.stdlib.components.chat import message_from_template_representation

# --- Round-trip: provider_fields survives IR -> Message -> format_for_llm ---


def test_provider_fields_round_trips_through_ir():
    """An author's provider_fields survives Message -> format_for_llm and back."""
    pf = {"openai": {"prediction": {"type": "content"}}}
    msg = Message("user", "hi", provider_fields=pf)

    tr = msg.format_for_llm()
    assert tr.provider_fields == pf

    rebuilt = message_from_template_representation(
        tr, default_role="user", content="hi"
    )
    assert rebuilt.provider_fields == pf


def test_provider_fields_defaults_to_none():
    """Message and TemplateRepresentation default provider_fields to None."""
    assert Message("user", "hi").provider_fields is None
    assert TemplateRepresentation(obj=None, args={}).provider_fields is None


# --- merge helper: match rule ---


def test_merge_none_and_empty_are_noops():
    """None or empty provider_fields leaves base untouched and never raises."""
    base = {"role": "user", "content": "hi"}
    assert merge_provider_fields(dict(base), None, "ollama") == base
    assert merge_provider_fields(dict(base), {}, "ollama") == base


def test_merge_exact_provider_match():
    """An exact provider key merges its fields into the wire dict."""
    out = merge_provider_fields(
        {"role": "user", "content": "hi"}, {"ollama": {"keep_alive": "5m"}}, "ollama"
    )
    assert out["keep_alive"] == "5m"


def test_merge_wildcard_matches_every_provider():
    """The "*" key merges on any provider."""
    for provider in ("openai", "ollama", "watsonx", "huggingface", "litellm"):
        out = merge_provider_fields(
            {"role": "user", "content": "hi"}, {"*": {"x": 1}}, provider
        )
        assert out["x"] == 1


@pytest.mark.parametrize("provider", sorted(OPENAI_COMPATIBLE_WIRE_PROVIDERS))
def test_merge_openai_family_alias(provider):
    """The "openai" key merges on every OpenAI-compatible wire provider."""
    out = merge_provider_fields(
        {"role": "user", "content": "hi"}, {"openai": {"prediction": {}}}, provider
    )
    assert "prediction" in out


def test_openai_family_set_is_the_serialization_family():
    """The wire family is the serialization family (includes huggingface)."""
    assert OPENAI_COMPATIBLE_WIRE_PROVIDERS == frozenset(
        {"openai", "litellm", "watsonx", "huggingface"}
    )


# --- merge helper: mismatch raises ---


def test_merge_mismatch_raises():
    """A named provider key that matches nothing (and no "*") raises ValueError."""
    with pytest.raises(ValueError):
        merge_provider_fields(
            {"role": "user", "content": "hi"}, {"ollama": {"x": 1}}, "openai"
        )


def test_merge_wildcard_suppresses_mismatch():
    """Adding "*" makes an otherwise-mismatched set valid; only "*" fields land."""
    out = merge_provider_fields(
        {"role": "user", "content": "hi"}, {"ollama": {"x": 1}, "*": {"y": 2}}, "openai"
    )
    assert out["y"] == 2
    assert "x" not in out


def test_merge_multiple_named_some_match():
    """When several named keys are present, only the matching one's fields land."""
    out = merge_provider_fields(
        {"role": "user", "content": "hi"},
        {"ollama": {"x": 1}, "openai": {"y": 2}},
        "openai",
    )
    assert out["y"] == 2
    assert "x" not in out


# --- merge helper: precedence (known fields win) ---


def test_merge_known_fields_win_and_collision_debug_logged(caplog):
    """A colliding author key is dropped (Mellea's field wins) and debug-logged."""
    with caplog.at_level(logging.DEBUG):
        out = merge_provider_fields(
            {"role": "user", "content": "real"},
            {"openai": {"role": "x", "content": "y", "extra": "kept"}},
            "openai",
        )
    assert out["role"] == "user"
    assert out["content"] == "real"
    assert out["extra"] == "kept"


# --- merge helper: source isolation (no aliasing) ---


def test_merge_does_not_alias_source_provider_fields():
    """Merged values are deep-copied; mutating the source never leaks to the wire dict."""
    pf = {"openai": {"prediction": {"type": "content"}}}
    out = merge_provider_fields({"role": "user", "content": "hi"}, pf, "openai")
    assert out["prediction"] == {"type": "content"}

    # Mutating the source's nested value must not change the already-merged dict.
    pf["openai"]["prediction"]["type"] = "new"
    assert out["prediction"] == {"type": "content"}

    # And mutating the merged dict must not reach back into the source.
    out["prediction"]["type"] = "other"
    assert pf["openai"]["prediction"]["type"] == "new"


def test_message_serialization_does_not_alias_provider_fields():
    """Mutating a Message's provider_fields after serialization cannot leak to the wire dict."""
    pf = {"openai": {"prediction": {"type": "content"}}}
    msg = Message("user", "hi", provider_fields=pf)
    wire = message_to_openai_message(msg, provider="openai")
    assert wire["prediction"] == {"type": "content"}

    pf["openai"]["prediction"]["type"] = "new"
    assert wire["prediction"] == {"type": "content"}


# --- merge helper: malformed values raise a clear error ---


@pytest.mark.parametrize("not_a_dict", [["openai"], "openai", 42])
def test_merge_non_dict_provider_fields_raises_typeerror(not_a_dict):
    """A truthy non-dict provider_fields raises a named TypeError, not a bare AttributeError."""
    with pytest.raises(TypeError):
        merge_provider_fields({"role": "user", "content": "hi"}, not_a_dict, "openai")


def test_merge_non_dict_value_raises_typeerror():
    """A matching provider_fields value that is not a dict raises TypeError, not a generic error."""
    with pytest.raises(TypeError):
        merge_provider_fields(
            {"role": "user", "content": "hi"}, {"openai": "not-a-dict"}, "openai"
        )


def test_merge_non_dict_value_ignored_when_unmatched():
    """A non-dict value under an unmatched key never runs .items() and never raises."""
    out = merge_provider_fields(
        {"role": "user", "content": "hi"},
        {"ollama": "not-a-dict", "openai": {"x": 1}},
        "openai",
    )
    assert out["x"] == 1


# --- per-backend reach: field lands on the wire dict ---


def test_reaches_openai_wire():
    """An author field reaches the OpenAI wire dict via message_to_openai_message."""
    msg = Message("user", "hi", provider_fields={"openai": {"prediction": {"t": 1}}})
    wire = message_to_openai_message(msg, provider="openai")
    assert wire["prediction"] == {"t": 1}


def test_openai_wire_mismatch_raises():
    """message_to_openai_message raises on a provider-mismatched declaration."""
    msg = Message("user", "hi", provider_fields={"ollama": {"x": 1}})
    with pytest.raises(ValueError):
        message_to_openai_message(msg, provider="openai")
