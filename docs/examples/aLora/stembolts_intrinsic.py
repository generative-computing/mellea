# type: ignore
"""Helper functions for loading and calling a fully custom, non-catalog adapter.

`stembolts` (Hugging Face: `nfulton/stembolts`) is a custom aLoRA adapter trained
against several base models (`base_model_name` is a constructor argument here,
not fixed) with its own output schema (`{"defective_part": str, "diag_likelihood":
float}`) rather than the generic `{"requirement_check": {"score": ...}}` shape
most catalog adapters use. It is invoked directly via `Intrinsic`/`mfuncs.act()`
and the raw JSON is returned to the caller — it is not a pass/fail
`ALoraRequirement` validator. Consumed by `102_example.py` in this directory.
"""

import mellea.stdlib.functional as mfuncs
from mellea.backends import Backend
from mellea.backends.adapters import (
    Adapter,
    AdapterMixin,
    Identity,
    LocalFileBinding,
    get_io_contract,
)
from mellea.backends.adapters.catalog import AdapterType
from mellea.core import Context
from mellea.stdlib.components import Message
from mellea.stdlib.components.intrinsic import Intrinsic

_INTRINSIC_MODEL_ID = "nfulton/stembolts"
_INTRINSIC_ADAPTER_NAME = "stembolts"
_INTRINSIC_QUALIFIED_NAME = f"{_INTRINSIC_ADAPTER_NAME}_{AdapterType.ALORA.value}"


def _stembolt_adapter() -> Adapter:
    """Compose the stembolts adapter from its Identity, output contract, and
    weights binding, rather than a dedicated shim class.

    `get_io_contract` falls back to a permissive dict contract for names
    outside the built-in catalog — sufficient here, since the adapter's real
    output shape (`{"defective_part": str, "diag_likelihood": float}`) is
    enforced by its own `io.yaml` on the Hugging Face repo, not by `IOContract`.
    """
    return Adapter(
        identity=Identity(name=_INTRINSIC_ADAPTER_NAME, adapter_type="alora"),
        io_contract=get_io_contract(_INTRINSIC_ADAPTER_NAME),
        weights=LocalFileBinding(
            name=_INTRINSIC_ADAPTER_NAME,
            adapter_type=AdapterType.ALORA,
            repo_id=_INTRINSIC_MODEL_ID,
            # A LocalFileBinding resolves revision=None via the adapter
            # function catalog, which has no entry for a custom, non-catalog
            # name like this one — pin explicitly instead.
            revision="main",
        ),
    )


class StemboltIntrinsic(Intrinsic):
    def __init__(self):
        # adapter_types is required here: "stembolts" is outside Mellea's
        # intrinsics catalog, so Intrinsic has no catalog entry to fall
        # back to (Epic #929, issue #1144).
        Intrinsic.__init__(
            self,
            intrinsic_name=_INTRINSIC_ADAPTER_NAME,
            adapter_types=(AdapterType.ALORA,),
        )


async def async_stembolt_failure_analysis(
    notes: str, ctx: Context, backend: Backend | AdapterMixin
):
    # add_adapter() refuses a duplicate qualified name with a warning, so guard first.
    if _INTRINSIC_QUALIFIED_NAME not in backend.list_adapters():
        backend.add_adapter(_stembolt_adapter())

    ctx = ctx.add(Message("user", content=notes))

    action = StemboltIntrinsic()
    mot, ctx = await backend.generate_from_context(action, ctx)
    return mot, ctx


def stembolt_failure_analysis(
    notes: str, ctx: Context, backend: Backend | AdapterMixin
):
    # add_adapter() refuses a duplicate qualified name with a warning, so guard first.
    if _INTRINSIC_QUALIFIED_NAME not in backend.list_adapters():
        backend.add_adapter(_stembolt_adapter())

    ctx = ctx.add(Message("user", content=notes))

    action = StemboltIntrinsic()
    return mfuncs.act(action, ctx, backend)
