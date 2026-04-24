import pytest
from mellea.backends.tools import MelleaTool
from mellea.core.base import ModelToolCall
from mellea.plugins import hook, HookType, PluginMode, block, register
from mellea.stdlib.functional import _acall_tools
from mellea.core.backend import Backend
from mellea.core.base import ModelOutputThunk, GenerateLog
from typing import Any

class _MockBackend(Backend):
    async def _generate_from_context(self, *args, **kwargs):
        pass
    async def generate_from_raw(self, *args, **kwargs):
        pass

@pytest.mark.asyncio
async def test_tool_is_internal_accessible_in_hook():
    """Verify that is_internal flag is accessible in TOOL_PRE_INVOKE hook."""
    
    observed_is_internal = []
    
    @hook(HookType.TOOL_PRE_INVOKE)
    async def checker(payload, _):
        observed_is_internal.append(payload.model_tool_call.func.is_internal)
        return None
    
    register(checker)
    
    # Create an internal tool
    internal_tool = MelleaTool.from_callable(lambda: "ok", name="internal", is_internal=True)
    external_tool = MelleaTool.from_callable(lambda: "ok", name="external", is_internal=False)
    
    tc_internal = ModelToolCall(name="internal", func=internal_tool, args={})
    tc_external = ModelToolCall(name="external", func=external_tool, args={})
    
    mot = ModelOutputThunk(value="", tool_calls={"internal": tc_internal, "external": tc_external})
    
    await _acall_tools(mot, _MockBackend())
    
    assert observed_is_internal == [True, False]

if __name__ == "__main__":
    pytest.main([__file__])
