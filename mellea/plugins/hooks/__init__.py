"""Hook payload classes for the Mellea plugin system."""

from mellea.plugins.hooks.component import (
    ComponentPostCreatePayload,
    ComponentPostErrorPayload,
    ComponentPostSuccessPayload,
    ComponentPreCreatePayload,
    ComponentPreExecutePayload,
)
from mellea.plugins.hooks.generation import (
    GenerationPostCallPayload,
    GenerationPreCallPayload,
    GenerationStreamChunkPayload,
)
from mellea.plugins.hooks.sampling import (
    SamplingIterationPayload,
    SamplingLoopEndPayload,
    SamplingLoopStartPayload,
    SamplingRepairPayload,
)
from mellea.plugins.hooks.session import (
    SessionCleanupPayload,
    SessionPostInitPayload,
    SessionPreInitPayload,
    SessionResetPayload,
)
from mellea.plugins.hooks.tool import ToolPostInvokePayload, ToolPreInvokePayload
from mellea.plugins.hooks.validation import (
    ValidationPostCheckPayload,
    ValidationPreCheckPayload,
)

__all__ = [
    # Component
    "ComponentPostCreatePayload",
    "ComponentPostErrorPayload",
    "ComponentPostSuccessPayload",
    "ComponentPreCreatePayload",
    "ComponentPreExecutePayload",
    # Generation
    "GenerationPostCallPayload",
    "GenerationPreCallPayload",
    "GenerationStreamChunkPayload",
    # Sampling
    "SamplingIterationPayload",
    "SamplingLoopEndPayload",
    "SamplingLoopStartPayload",
    "SamplingRepairPayload",
    # Session
    "SessionCleanupPayload",
    "SessionPostInitPayload",
    "SessionPreInitPayload",
    "SessionResetPayload",
    # Tool
    "ToolPostInvokePayload",
    "ToolPreInvokePayload",
    # Validation
    "ValidationPostCheckPayload",
    "ValidationPreCheckPayload",
]
