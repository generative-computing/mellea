# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Library-backed helper for reading LLMON replies.

When an LLMON model is served behind a chat-completions endpoint, its
`chat_template.jinja` does the llmonizing server-side: Mellea sends a message and
the server draws the block ids, emits the blocks, and wires the `exec` graph. (The
exact outgoing shape — a flat `{role, content}` string vs. a dict carrying
structured `instruction`/`data`/`exec` fields — is an unresolved design question;
see `LLMON-STATUS.md`. It does not affect this reply-parsing helper.) The reply
comes back in the assistant message, and depending on the server configuration its
`content` is either detokenized prose or raw Machine LLMON
(`<|open|>assistant<|:|>...<|close|> ... <|open_end|>`).

`parse_llmon_reply` handles both: it detects the Machine-LLMON open token and, if
present, parses the reply with the `llmon` library and extracts the assistant
block's text; otherwise it returns the reply unchanged. This is the single
primitive `LLMONInstruction._parse` delegates to.

The conversion is delegated to the `llmon` library (arXiv:2603.22519), an
optional dependency installed via `pip install "mellea[llmon]"`.
"""

from __future__ import annotations

import json

try:
    from converters.common import common as _llmon_common
    from converters.JSON.llmon_converters import LLMONConverters
except ImportError as e:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "LLMON support requires the `llmon` library. "
        'Please install it with: pip install "mellea[llmon]"'
    ) from e

# The Machine-LLMON opening token; its presence in a reply signals raw LLMON.
_OPEN_TAG: str = _llmon_common.OPEN_TAG
# The library preserves the raw `<|:|>` separator in JSON keys (e.g.
# "assistant<|:|>1234"), so match assistant blocks against this prefix.
_ASSISTANT_KEY_PREFIX: str = "assistant" + _llmon_common.COLON_SEPARATOR_TAG


def is_llmon(text: str) -> bool:
    """Return whether `text` contains raw Machine-LLMON markup.

    Args:
        text: The reply text to inspect.

    Returns:
        `True` if the Machine-LLMON open token is present, else `False`.
    """
    return _OPEN_TAG in text


def parse_llmon_reply(text: str) -> str:
    """Extract the assistant text from a (possibly) Machine-LLMON reply.

    If `text` contains raw Machine LLMON, it is parsed with the `llmon` library
    and the assistant block's text is returned. If `text` is plain prose (the
    server already detokenized), or if parsing fails, `text` is returned
    unchanged.

    Args:
        text: The assistant message content returned by the served model.

    Returns:
        The assistant text, with any Machine-LLMON markup stripped.
    """
    if not is_llmon(text):
        return text

    converter = LLMONConverters(converting_to_json=True)
    try:
        json_str = converter.mllmon_str_to_json_str(text)
        parsed = json.loads(json_str)
    except Exception:
        # Robustness: a partial or non-conforming reply is treated as plain text
        # rather than raising, mirroring the soft-mode passthrough contract.
        return text

    return _extract_assistant_text(parsed, fallback=text)


def _extract_assistant_text(parsed: object, *, fallback: str) -> str:
    """Pull the assistant block's text out of the library's parsed JSON.

    The library renders an `assistant:<id>` block as a one-key object whose key
    keeps the raw separator, e.g. `{"assistant<|:|>1234": "..."}`. Prefer that
    key; else fall back to the sole value of a single-key object; else `fallback`.

    Args:
        parsed: The Python object `json.loads` produced from the library output.
        fallback: Text to return when no assistant text can be located.

    Returns:
        The extracted assistant text, or `fallback`.
    """
    if isinstance(parsed, dict):
        for key, value in parsed.items():
            if isinstance(key, str) and (
                key == "assistant" or key.startswith(_ASSISTANT_KEY_PREFIX)
            ):
                return value if isinstance(value, str) else str(value)
        # No assistant key: a single-key object is almost certainly the reply.
        if len(parsed) == 1:
            (value,) = parsed.values()
            return value if isinstance(value, str) else str(value)
    elif isinstance(parsed, str):
        return parsed
    return fallback
