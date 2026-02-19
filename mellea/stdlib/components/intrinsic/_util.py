"""Shared utilities for intrinsic convenience wrappers."""

import json

from ....backends import ModelOption
from ....backends.adapters import AdapterMixin, AdapterType, GraniteCommonAdapter
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
    adapter = GraniteCommonAdapter(
        intrinsic_name, adapter_type=AdapterType.LORA, base_model_name=base_model_name
    )
    if adapter.qualified_name not in backend.list_adapters():
        backend.add_adapter(adapter)

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
