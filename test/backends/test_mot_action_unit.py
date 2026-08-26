# Copyright 2024 Mellea Contributors
# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for passing a computed `ModelOutputThunk` as the originating action.

Covers psschwei's Step 2 of issue #1030: when a computed `ModelOutputThunk` is
reused as the next incoming turn, its `parsed_repr` must be re-parsed into a rich
`Message` (preserving role/content/tool_calls/thinking) rather than degrading to
the raw string value.
"""

from mellea.core import ModelOutputThunk, RawProviderResponse
from mellea.core.backend import generate_walk
from mellea.stdlib.components import Message


def _computed_mot_action(
    role: str = "assistant", content: str = "hi"
) -> ModelOutputThunk:
    """Build a computed MOT with a provider-native response, set as its own action."""
    mot = ModelOutputThunk(value=content)
    mot.raw = RawProviderResponse(
        provider="openai",
        response={"choices": [{"message": {"role": role, "content": content}}]},
    )
    mot._call.action = mot  # a computed MOT reused as the incoming turn
    return mot


def test_computed_mot_action_parses_to_message_not_string():
    """A computed-MOT action re-parses into a rich `Message`, not a raw string."""
    mot = _computed_mot_action(role="assistant", content="the answer")

    mot._set_parsed_repr()

    assert isinstance(mot.parsed_repr, Message), (
        f"expected a Message, got {type(mot.parsed_repr).__name__}"
    )
    assert mot.parsed_repr.role == "assistant"
    assert mot.parsed_repr.content == "the answer"


def test_computed_mot_action_does_not_self_generate():
    """A computed MOT must not be re-generated when reused as an action.

    `generate_walk` returns only *uncomputed* MOT leaves; a computed MOT yields
    an empty list, so reusing one as the incoming turn triggers no spurious
    model call.
    """
    mot = _computed_mot_action()
    assert mot.is_computed()
    assert generate_walk(mot) == []
