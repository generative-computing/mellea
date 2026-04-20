"""Shared utilities for intrinsic convenience wrappers."""

import json

from ....backends import ModelOption
from ....backends.adapters import (
    AdapterMixin,
    AdapterType,
    EmbeddedIntrinsicAdapter,
    IntrinsicAdapter,
)
from ....stdlib import functional as mfuncs
from ...context import ChatContext
from .intrinsic import Intrinsic


def call_intrinsic(
    intrinsic_name: str,
    context: ChatContext,
    backend: AdapterMixin,
    /,
    kwargs: dict | None = None,
):
    """Shared code for invoking intrinsics.

    :returns: Result of the call in JSON format.
    """
    # Adapter needs to be present in the backend before it can be invoked.
    # We must create the Adapter object in order to determine whether we need to create
    # the Adapter object.
    base_model_name = backend.base_model_name
    if base_model_name is None:
        raise ValueError("Backend has no model ID")

    # Check if the backend already has the adapter.
    has_adapter = any(
        qualified_name.startswith(f"{intrinsic_name}_")
        for qualified_name in backend.list_adapters()
    )

    # TODO: We should improve this logic. For now, we know that there are two cases of
    # adapter loading: 1. regular adapters, and 2. embedded adapters.
    if not has_adapter:
        # EmbeddedAdapters get grabbed directly from the hf repo.
        if getattr(backend, "_uses_embedded_adapters", False):
            repo_id = getattr(backend, "_model_id", backend.base_model_name)
            adapters = EmbeddedIntrinsicAdapter.from_hub(
                repo_id, intrinsic_name=intrinsic_name
            )
            # Only one adapter should be returned, but we add any returned here in case.
            for adapter in adapters:
                backend.add_adapter(adapter)
        else:
            # Regular IntrinsicAdapters utilize a catalog to download during their instantiation.
            intrinsic_adapter = IntrinsicAdapter(
                intrinsic_name,
                adapter_type=AdapterType.LORA,
                base_model_name=base_model_name,
            )
            backend.add_adapter(intrinsic_adapter)

    # Create the AST node for the action we wish to perform.
    intrinsic = Intrinsic(intrinsic_name, intrinsic_kwargs=kwargs)

    # Execute the AST node.
    model_output_thunk, _ = mfuncs.act(
        intrinsic,
        context,
        backend,
        model_options={ModelOption.TEMPERATURE: 0.0},
        # No rejection sampling, please
        strategy=None,
    )

    # act() can return a future. Don't know how to handle one from non-async code.
    assert model_output_thunk.is_computed()

    # Output of an Intrinsic action is the string representation of the output of the
    # intrinsic. Parse the string.
    result_str = model_output_thunk.value
    if result_str is None:
        raise ValueError("Model output is None.")
    result_json = json.loads(result_str)
    return result_json
