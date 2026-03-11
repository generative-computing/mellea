"""Hook payload classes for the Mellea plugin system."""

from .component import (
    ComponentPostErrorPayload,
    ComponentPostSuccessPayload,
    ComponentPreExecutePayload,
)
from .generation import GenerationPostCallPayload, GenerationPreCallPayload
from .sampling import (
    SamplingIterationPayload,
    SamplingLoopEndPayload,
    SamplingLoopStartPayload,
    SamplingRepairPayload,
)
from .session import (
    SessionCleanupPayload,
    SessionPostInitPayload,
    SessionPreInitPayload,
    SessionResetPayload,
)
from .tool import ToolPostInvokePayload, ToolPreInvokePayload
from .validation import ValidationPostCheckPayload, ValidationPreCheckPayload

__all__ = [
    # Component
    "ComponentPostErrorPayload",
    "ComponentPostSuccessPayload",
    "ComponentPreExecutePayload",
    # Generation
    "GenerationPostCallPayload",
    "GenerationPreCallPayload",
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
