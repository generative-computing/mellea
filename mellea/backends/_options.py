# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared model-options resolution for backends and adapter-function helpers."""

from typing import Any

from .model_options import ModelOption


def resolve_model_options(
    *,
    backend_defaults: dict[str, Any],
    remap: dict[str, str],
    call_options: dict[str, Any] | None,
    helper_defaults: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve the final model options for a generation call.

    Precedence, lowest to highest: *backend_defaults* < *helper_defaults* <
    *call_options*. *remap* is applied to *backend_defaults* and
    *call_options* to translate backend/caller-specific option names to
    Mellea's `ModelOption` keys; *helper_defaults* is assumed to already be
    in `ModelOption` form and is not remapped.

    Args:
        backend_defaults (dict[str, Any]): The backend's own default model
            options, in backend-specific or Mellea key form.
        remap (dict[str, str]): Mapping from backend/caller-specific option
            names to `ModelOption` keys, as consumed by
            `ModelOption.replace_keys`.
        call_options (dict[str, Any] | None): Per-call model options supplied
            by the caller. `None` is treated as empty.
        helper_defaults (dict[str, Any] | None): Defaults supplied by a
            higher-level helper (e.g. an adapter-function wrapper), already
            in `ModelOption` key form. `None` is treated as empty.

    Returns:
        dict[str, Any]: A new merged dictionary of model options.
    """
    backend_opts = ModelOption.replace_keys(backend_defaults, remap)
    call_opts = ModelOption.replace_keys(call_options, remap) if call_options else {}

    merged = ModelOption.merge_model_options(backend_opts, helper_defaults)
    return ModelOption.merge_model_options(merged, call_opts)
