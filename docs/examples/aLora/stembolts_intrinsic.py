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
from mellea.backends.adapters import AdapterMixin
from mellea.backends.adapters.adapter import CustomIntrinsicAdapter
from mellea.core import Context
from mellea.stdlib.components import Message
from mellea.stdlib.components.intrinsic import Intrinsic

_INTRINSIC_MODEL_ID = "nfulton/stembolts"
_INTRINSIC_ADAPTER_NAME = "stembolts"


class StemboltAdapter(CustomIntrinsicAdapter):
    def __init__(self, base_model_name: str):
        super().__init__(
            model_id=_INTRINSIC_MODEL_ID,
            intrinsic_name=_INTRINSIC_ADAPTER_NAME,
            base_model_name=base_model_name,
        )


class StemboltIntrinsic(Intrinsic):
    def __init__(self):
        Intrinsic.__init__(self, intrinsic_name=_INTRINSIC_ADAPTER_NAME)


async def async_stembolt_failure_analysis(
    notes: str, ctx: Context, backend: Backend | AdapterMixin
):
    # add_adapter() refuses a duplicate qualified name with a warning, so guard first.
    adapter = StemboltAdapter(backend.base_model_name)
    if adapter.qualified_name not in backend.list_adapters():
        backend.add_adapter(adapter)

    ctx = ctx.add(Message("user", content=notes))

    action = StemboltIntrinsic()
    mot, ctx = await backend.generate_from_context(action, ctx)
    return mot, ctx


def stembolt_failure_analysis(
    notes: str, ctx: Context, backend: Backend | AdapterMixin
):
    # add_adapter() refuses a duplicate qualified name with a warning, so guard first.
    adapter = StemboltAdapter(backend.base_model_name)
    if adapter.qualified_name not in backend.list_adapters():
        backend.add_adapter(adapter)

    ctx = ctx.add(Message("user", content=notes))

    action = StemboltIntrinsic()
    return mfuncs.act(action, ctx, backend)
