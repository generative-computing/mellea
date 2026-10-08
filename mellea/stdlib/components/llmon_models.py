# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Import-safe allowlist of served models that speak LLMON.

This module names the served models for which Mellea automatically encodes input
into Machine LLMON and strips LLMON out of replies (see `llmon.py` for the actual
encode/parse helpers). It is deliberately **dependency-free**: unlike `llmon.py`,
it does not import the optional `llmon` library, so `mellea.core` can import it to
gate the two LLMON seams (`ModelOutputThunk._set_parsed_repr` and
`Backend.generate_from_context`) without requiring the `mellea[llmon]` extra. The
heavy lifting (`parse_llmon_reply`, `encode_llmon_input`, `encode_instruction_for_llmon`,
`llmon_grammar`) stays in `llmon.py` and is imported lazily only once a model here
has matched.

Membership is an exact string match against the backend's resolved model id. Add a
served LLMON model by adding its id to `_LLMON_MODELS`.
"""

from __future__ import annotations

# Served model ids whose weights emit/consume Machine LLMON. Seeded with the one
# model verified to date (see `LLMON-SUMMARY.md`). Exact-match on the resolved id.
_LLMON_MODELS: frozenset[str] = frozenset({"llmon95:latest"})


def is_llmon_model(model_id: str | None) -> bool:
    """Return whether a served model id is a known LLMON model.

    Args:
        model_id: The backend's resolved model id (e.g. the value a backend stamps
            onto `mot.generation.model`, or its `self._model_id`). `None` when no
            model is known yet.

    Returns:
        `True` if `model_id` is in the LLMON allowlist, else `False`. `None` is
        never a match.
    """
    if model_id is None:
        return False
    return model_id in _LLMON_MODELS
