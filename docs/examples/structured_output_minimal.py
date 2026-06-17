# pytest: ollama, e2e

"""Minimal example: structured output via act(format=) with a Pydantic model."""

from pydantic import BaseModel

from mellea import start_session
from mellea.stdlib.components import Instruction


class Classification(BaseModel):
    label: str
    confidence: float


with start_session("ollama") as m:
    result = m.act(
        Instruction(
            description="Classify the sentiment of 'ship it!' and provide a confidence score."
        ),
        format=Classification,
    )
    # result.value is a JSON string, not a Classification instance
    parsed = Classification.model_validate_json(str(result))
    print(parsed.label, parsed.confidence)
