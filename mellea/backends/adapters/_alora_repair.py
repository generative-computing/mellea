# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Repair for published aLoRA io.yaml files that cannot activate their adapter.

A removable workaround for issue #1679 (the published `requirement-check`
aLoRA, all granite-4.1 slots): its io.yaml instruction does not tokenise to the
adapter's declared `alora_invocation_tokens`. `LocalHFBackend.add_adapter`
applies it through `LocalHFBackend._repair_alora_instruction`. Once the
publisher republishes the adapter and the catalogue pin is bumped, delete this
module, that method, and its two call sites in `add_adapter`.
"""

from __future__ import annotations

import json
import pathlib
from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase


def _token_sequence_present(
    tokenizer: PreTrainedTokenizerBase, text: str, token_ids: Sequence[int]
) -> bool:
    """Whether `token_ids` occurs as a contiguous run in `tokenizer.encode(text)`.

    Args:
        tokenizer: The tokenizer to encode with.
        text: The text to search.
        token_ids: The token run to look for.

    Returns:
        True if the run occurs at least once in the encoded text.
    """
    if not token_ids:
        return False
    tokens = tokenizer.encode(text, add_special_tokens=False)
    seq = list(token_ids)
    n = len(seq)
    return any(tokens[i : i + n] == seq for i in range(len(tokens) - n + 1))


def _read_alora_invocation_tokens(adapter_dir: str) -> list[int] | None:
    """Read the declared `alora_invocation_tokens` from a downloaded adapter.

    Args:
        adapter_dir: Local directory holding the adapter's
            `adapter_config.json`.

    Returns:
        The declared invocation token ids, or `None` when the directory has
        no `adapter_config.json` or the config declares none (a plain LoRA).
    """
    config_path = pathlib.Path(adapter_dir) / "adapter_config.json"
    if not config_path.is_file():
        return None
    tokens = json.loads(config_path.read_text(encoding="utf-8")).get(
        "alora_invocation_tokens"
    )
    return list(tokens) if tokens else None


def _alora_invocation_repair(
    tokenizer: PreTrainedTokenizerBase,
    instruction: str,
    invocation_tokens: Sequence[int],
) -> str | None:
    """Repair an aLoRA io.yaml instruction that cannot activate its own adapter.

    An aLoRA adapter activates only when its declared `alora_invocation_tokens`
    occur, after tokenisation, in the assembled prompt. Some published
    adapters (issue #1679: `requirement-check`, all granite-4.1 slots) ship an
    instruction whose text does not tokenise to the declared sequence, because
    the Granite tokeniser merges `>` with a following `:` into one token, so
    the instruction's `<requirements>:` never yields the declared
    `<requirements>` run.

    The repair is deliberately verification-driven and self-terminating: it
    returns a changed instruction only when the declared sequence is absent
    from the tokenised instruction, the decoded invocation text followed by a
    single colon is present in it, and dropping that colon makes the declared
    sequence present. A correctly republished file passes the first check and
    is left untouched, in either direction a publisher might fix it (the
    instruction text changed, or the declared tokens changed to match the
    existing text).

    Args:
        tokenizer: The base model's tokenizer.
        instruction: The io.yaml `instruction` template text.
        invocation_tokens: The adapter's declared `alora_invocation_tokens`.

    Returns:
        The repaired instruction, or `None` when the instruction is already
        consistent with the declared sequence or no repair is possible.
    """
    if _token_sequence_present(tokenizer, instruction, invocation_tokens):
        return None
    invocation_text = cast(
        str, tokenizer.decode(list(invocation_tokens), skip_special_tokens=False)
    )
    broken = invocation_text + ":"
    if broken not in instruction:
        return None
    repaired = instruction.replace(broken, invocation_text, 1)
    if _token_sequence_present(tokenizer, repaired, invocation_tokens):
        return repaired
    return None
