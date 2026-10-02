# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load a cached Granite tokenizer, so tests can render the real chat template without torch or model weights."""

import pytest

_GRANITE_THINKING_MODEL_ID = "ibm-granite/granite-4.2-3b"


def _try_load_granite_tokenizer(model_id: str):
    """Return a Granite tokenizer from the local cache, or None.

    A cached `config.json` does not guarantee the tokenizer's own files
    (`tokenizer.json`, etc.) are also cached — e.g. after an interrupted or
    partial download. `local_files_only=True` raises `OSError` in that case;
    treat it the same as an absent cache rather than letting the test error.
    Skips the calling test if `transformers` (the `mellea[hf]` extra) is missing.
    """
    pytest.importorskip("transformers", reason="transformers not installed")
    from huggingface_hub import _CACHED_NO_EXIST, try_to_load_from_cache
    from transformers import AutoTokenizer

    cached_config = try_to_load_from_cache(model_id, "config.json")
    if cached_config is None or cached_config is _CACHED_NO_EXIST:
        return None
    try:
        return AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    except OSError:
        return None
