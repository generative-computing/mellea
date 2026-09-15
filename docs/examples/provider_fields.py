# pytest: unit
"""Emit a provider-specific wire field without a core-type change.

A component author can attach `provider_fields` to a `Message` to declare extra
keys that Mellea does not model, targeted at a specific provider. The fields are
merged into the wire message at serialization time. Mellea's known fields always
win on a collision, and a declaration that names a provider the request never
reaches raises a `ValueError` (add `"*"` to opt out of that check).

See issue #1565 for the design.
"""

import pytest

from mellea.helpers.openai_compatible_helpers import message_to_openai_message
from mellea.stdlib.components import Message


def targeted_provider_field() -> dict:
    """Attach an OpenAI-only `prediction` field and serialize for OpenAI.

    The `"openai"` key matches the OpenAI wire family (openai, litellm, watsonx,
    huggingface), so the field lands on the wire message for any of them.
    """
    msg = Message(
        "user",
        "Refactor this function.",
        provider_fields={"openai": {"prediction": {"type": "content"}}},
    )
    wire = message_to_openai_message(msg, provider="openai")
    assert wire["prediction"] == {"type": "content"}
    return wire


def portable_field_with_wildcard() -> dict:
    """Use `"*"` to declare a field valid on every backend.

    A `"*"` target never raises on a provider mismatch — it is the author's
    portability contract that the field is safe to send everywhere.
    """
    msg = Message("user", "Hello", provider_fields={"*": {"metadata_tag": "demo"}})
    wire = message_to_openai_message(msg, provider="openai")
    assert wire["metadata_tag"] == "demo"
    return wire


def known_fields_always_win() -> dict:
    """An author key that collides with a Mellea-known field is dropped."""
    msg = Message(
        "user",
        "real content",
        provider_fields={"openai": {"content": "hijacked", "extra": "kept"}},
    )
    wire = message_to_openai_message(msg, provider="openai")
    assert wire["content"] == "real content"  # Mellea's field wins
    assert wire["extra"] == "kept"  # non-colliding author field lands
    return wire


def provider_mismatch_raises() -> None:
    """Targeting a provider the request never reaches is a hard error."""
    msg = Message("user", "Hi", provider_fields={"ollama": {"keep_alive": "5m"}})
    with pytest.raises(ValueError):
        message_to_openai_message(msg, provider="openai")


if __name__ == "__main__":
    print("Targeted field:", targeted_provider_field())
    print("Wildcard field:", portable_field_with_wildcard())
    print("Known fields win:", known_fields_always_win())
    provider_mismatch_raises()
    print("Provider mismatch raised as expected.")
