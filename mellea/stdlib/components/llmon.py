# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Library-backed helpers for the LLMON round-trip.

A model trained on LLMON (arXiv:2603.22519) emits Machine LLMON directly: the
reply's `content` is either plain prose or raw Machine LLMON
(`<|open|>assistant<|:|>...<|close|> ... <|open_end|>`). Whether anything LLMONizes
the *input* is deployment-dependent — the tested `llmon95:latest` Ollama model has
a passthrough chat template and relies on the weights, so Mellea owns both
directions if structured input is wanted. See `LLMON-SUMMARY.md`.

This module provides the two primitives:

- `parse_llmon_reply` (reply-in): detects the Machine-LLMON open token and, if
  present, extracts the assistant block's plain text; otherwise returns the reply
  unchanged. Robust to the partial/garbled fragments a served model returns. This
  is what `LLMONInstruction._parse` delegates to.
- `encode_llmon_input` (out-direction): encodes an instruction (and optional data)
  into a Machine-LLMON `instr`/`data` object, the structured input the model was
  trained to read. This is what `LLMONInstruction.format_for_llm` uses when
  `llmonize_input=True`.

Both delegate the actual grammar work to the `llmon` library, an optional
dependency installed via `pip install "mellea[llmon]"`.
"""

from __future__ import annotations

import json
import re

try:
    from lang.common import common as _llmon_common
    from lang.json.llmon_converters import LLMONConverters
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

# A well-formed assistant block, used by the lenient fallback when the library's
# strict whole-reply conversion fails on a partial/garbled generation. A served
# model whose chat template consumed the opening blocks can return a fragment
# that begins mid-stream, so the answer sits in a locatable
# `<|open|>[data<|:|>]assistant<|:|>ID<|close|> TEXT <|open_end|>...` block rather
# than a document the converter can parse whole. The capture group is the block's
# text; the final such block is the answer (earlier ones are echoed context).
_ASSISTANT_BLOCK_RE: re.Pattern[str] = re.compile(
    re.escape(_llmon_common.OPEN_TAG)
    + r"(?:data"
    + re.escape(_llmon_common.COLON_SEPARATOR_TAG)
    + r")?"
    + r"assistant"
    + re.escape(_llmon_common.COLON_SEPARATOR_TAG)
    + r"\d+"
    + re.escape(_llmon_common.CLOSE_TAG)
    + r"(.*?)"
    + re.escape(_llmon_common.OPEN_END_TAG),
    re.DOTALL,
)


def is_llmon(text: str) -> bool:
    """Return whether `text` contains raw Machine-LLMON markup.

    Args:
        text: The reply text to inspect.

    Returns:
        `True` if the Machine-LLMON open token is present, else `False`.
    """
    return _OPEN_TAG in text


def encode_llmon_input(instruction: str, data: str | None = None) -> str:
    """Encode an instruction (and optional data) as a Machine-LLMON input object.

    Produces the structured input an LLMON-trained model reads: an object with an
    `instr` member carrying the instruction and, when `data` is given, a `data`
    member carrying the content the instruction operates on. This is the
    out-direction counterpart to `parse_llmon_reply` — it builds what Mellea
    *sends*, so the model receives explicit instruction/data boundaries rather
    than having to infer them from a plain prompt.

    The returned string is Machine LLMON (special tokens), not Human (backslash)
    LLMON, since Machine LLMON is what is sent to the model. The encoding is
    delegated to the `llmon` library and round-trips back through
    `parse_llmon_reply`'s converter.

    Args:
        instruction: The rendered instruction text.
        data: Optional content the instruction operates on (e.g. the text to
            summarise). Omitted from the object when `None`.

    Returns:
        A compact Machine-LLMON object string, e.g.
        `<|open|>object<|close|><|open|>instr<|close|>…<|open_end|>instr<|close|>…`.
    """
    obj: dict[str, str] = {"instr": instruction}
    if data is not None:
        obj["data"] = data
    # Pre-stringified plain-`str` values, so the default `with_mellea=False` is
    # correct here. The library's `with_mellea=True` only changes which *input*
    # types the converter accepts — it lets a dict value be a Mellea `CBlock`
    # (unwrapping `.value`) instead of raising `TypeError` — and the encoded output
    # is byte-identical either way. Passing plain strings keeps this helper a pure
    # `(str, str | None) -> str` with no `llmon → mellea.core` back-import.
    converter = LLMONConverters(converting_to_json=False)
    return converter.json_dict_to_llmon_str(obj, compact=True)


def parse_llmon_reply(text: str) -> str:
    """Extract the assistant text from a (possibly) Machine-LLMON reply.

    If `text` contains raw Machine LLMON, the assistant block's text is extracted.
    If `text` is plain prose (the server already detokenized), it is returned
    unchanged. Extraction is tried in two steps:

    1. The `llmon` library converts the whole reply to JSON and the assistant
       block is read from it. This is the clean path for a well-formed document.
    2. If step 1 raises — the library is strict and a served model can return a
       *partial* fragment (it begins mid-stream, carries unescaped control
       characters, or echoes instructions as preamble) — the reply is scanned
       directly for the last well-formed assistant block and that block's text is
       returned. The final block is the answer; earlier blocks are echoed context.

    If neither step finds an assistant block (e.g. a fragment with no complete
    block at all), `text` is returned unchanged rather than mangled — a blanket
    token strip would leave id-triple debris, so a reply with no recoverable block
    degrades to raw text rather than to garbage.

    Args:
        text: The assistant message content returned by the served model.

    Returns:
        The assistant text, with any Machine-LLMON markup stripped when a block is
        recoverable, else `text` unchanged.
    """
    if not is_llmon(text):
        return text

    converter = LLMONConverters(converting_to_json=True)
    try:
        json_str = converter.llmon_str_to_json_str(text)
        parsed = json.loads(json_str)
    except Exception:
        # The strict converter failed on a partial/garbled reply. Fall back to a
        # direct scan for the assistant block rather than returning raw markup.
        return _extract_assistant_block(text, fallback=text)

    return _extract_assistant_text(parsed, fallback=text)


def _extract_assistant_block(text: str, *, fallback: str) -> str:
    """Scan raw Machine-LLMON for the last assistant block and return its text.

    The lenient fallback for when the library's whole-reply conversion fails on a
    partial generation. Returns the text of the final
    `<|open|>[data<|:|>]assistant<|:|>ID<|close|> … <|open_end|>` block (the
    answer; earlier blocks are echoed context), or `fallback` when no complete
    block is present.

    Args:
        text: The raw reply, known to contain the Machine-LLMON open token.
        fallback: Text to return when no complete assistant block is found.

    Returns:
        The last assistant block's text, or `fallback`.
    """
    matches = _ASSISTANT_BLOCK_RE.findall(text)
    if matches:
        return matches[-1]
    return fallback


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
