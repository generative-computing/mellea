# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend-neutral helpers for inspecting assembled chat conversations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..stdlib.components import Message


def has_user_content(messages: list[Message]) -> bool:
    """Whether the assembled conversation carries real user-role content.

    Guards against sending a chat request whose user turns are all empty:
    `SimpleContext` intentionally discards recorded turns from
    `view_for_generation()`, so a caller who chains `.add(...)` and then
    passes an empty action would otherwise hit the model with no user-role
    content at all. Some chat models spin on empty prompts and burn tokens
    silently. Whitespace-only text counts as empty; images, audio, and
    documents count as content.

    Args:
        messages: The chat messages about to be sent to the model.

    Returns:
        True if any user-role message has non-whitespace text, images,
        audio, or documents; False otherwise.
    """
    return any(
        m.role == "user"
        and (
            (m.content and m.content.strip())
            or m.images
            or m.audio
            or getattr(m, "_docs", None)
        )
        for m in messages
    )
