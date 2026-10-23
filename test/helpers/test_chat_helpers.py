# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for mellea.helpers.chat_helpers."""

import pytest

from mellea.core.base import AudioBlock, ImageBlock
from mellea.helpers.chat_helpers import has_user_content
from mellea.stdlib.components import Message

# 1x1 transparent PNG; the guard only checks presence, not decodability.
_B64_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "YAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)


def test_empty_conversation_has_no_user_content():
    """An empty message list carries no user content."""
    assert not has_user_content([])


def test_user_text_is_content():
    """A user message with non-whitespace text counts as content."""
    assert has_user_content([Message("user", "real question")])


@pytest.mark.parametrize("text", ["", "   ", "\n\t  \n"])
def test_blank_user_text_is_not_content(text):
    """Empty and whitespace-only user text counts as empty."""
    assert not has_user_content([Message("user", text)])


@pytest.mark.parametrize("role", ["system", "assistant", "tool"])
def test_non_user_roles_are_ignored(role):
    """Content on non-user roles does not satisfy the guard."""
    assert not has_user_content([Message(role, "real content")])


def test_blank_user_text_after_real_assistant_turn_is_not_content():
    """The guard keys on the user role, not on any message having text."""
    assert not has_user_content(
        [Message("assistant", "an earlier real answer"), Message("user", "   ")]
    )


def test_user_text_after_blank_user_turn_is_content():
    """Any single user turn with real text satisfies the guard."""
    assert has_user_content(
        [
            Message("user", ""),
            Message("assistant", "placeholder"),
            Message("user", "the real question"),
        ]
    )


def test_user_images_count_as_content():
    """A user message with empty text but images counts as content."""
    msg = Message("user", "", images=[ImageBlock(value=_B64_PNG)])
    assert has_user_content([msg])


def test_user_audio_counts_as_content():
    """A user message with empty text but audio counts as content."""
    msg = Message("user", "", audio=[AudioBlock("UklGRg==", format="wav")])
    assert has_user_content([msg])


def test_user_documents_count_as_content():
    """A user message with empty text but documents counts as content."""
    msg = Message("user", "", documents=["product spec passage"])
    assert has_user_content([msg])
