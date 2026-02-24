# type: ignore
# pytest: skip, huggingface, requires_heavy_ram, llm
# SKIP REASON: needs to update.

import mellea.stdlib.functional as mfuncs
from mellea.backends import Backend
from mellea.backends.adapters import AdapterMixin
from mellea.backends.adapters.adapter import CustomGraniteCommonAdapter
from mellea.core import Context
from mellea.stdlib.components import Message
from mellea.stdlib.components.intrinsic import Intrinsic
from mellea.stdlib.components.simple import SimpleComponent

_INTRINSIC_MODEL_ID = "nfulton/stembolts"
_INTRINSIC_ADAPTER_NAME = "stembolts"


class StemboltAdapter(CustomGraniteCommonAdapter):
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
    # Backend.add_adapter should be idempotent, but we'll go ahead and check just in case.
    if _INTRINSIC_ADAPTER_NAME not in backend.list_adapters():
        backend.add_adapter(StemboltAdapter(backend.base_model_name))

    ctx = ctx.add(Message("user", content=notes))

    action = StemboltIntrinsic()
    mot = await backend.generate_from_context(action, ctx)
    return mot


def stembolt_failure_analysis(
    notes: str, ctx: Context, backend: Backend | AdapterMixin
):
    # Backend.add_adapter should be idempotent, but we'll go ahead and check just in case.
    adapter = StemboltAdapter(backend.base_model_name)
    if adapter.qualified_name not in backend.list_adapters():
        backend.add_adapter(adapter)

    ctx = ctx.add(Message("user", content=notes))

    action = StemboltIntrinsic()
    return mfuncs.act(action, ctx, backend)
