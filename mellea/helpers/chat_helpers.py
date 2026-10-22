# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend-neutral helpers for chat-message handling."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..stdlib.components import Message


def has_user_content(messages: list[Message]) -> bool:
    """Whether the assembled conversation carries real user-role content.

    Whitespace-only text counts as empty; images, audio, and documents
    count as content.

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
