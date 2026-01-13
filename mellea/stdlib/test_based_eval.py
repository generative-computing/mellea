"""LLM Evaluation with Unit Tests in Mellea."""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from mellea.security import SecLevel
from mellea.stdlib.base import (
    CBlock,
    Component,
    ModelOutputThunk,
    TemplateRepresentation,
)


class Message(BaseModel):
    """Schema for a message in the test data."""

    role: str
    content: str


class Example(BaseModel):
    """Schema for an example in the test data."""

    input: list[Message]
    targets: list[Message] = Field(default_factory=list)
    input_id: str = ""


class TestData(BaseModel):
    """Schema for test data loaded from json."""

    source: str
    name: str
    instructions: str
    examples: list[Example] = Field(default_factory=list)
    id: str

    @field_validator("examples")
    @classmethod
    def validate_examples(cls, v):
        """Ensure examples list is not empty."""
        if not v:
            raise ValueError("examples list cannot be empty")
        return v


class TestBasedEval(Component[str]):
    """Each TestBasedEval represents a single unit test."""

    def __init__(
        self,
        source: str,
        name: str,
        instructions: str,
        inputs: list[str],
        targets: list[list[str]] | None = None,  # can be optional
        test_id: str | None = None,
        input_ids: list[str] | None = None,
    ):
        """Initialize TestBasedEval (for a single unit test)."""
        self.source = source
        self.name = name
        self.instructions = instructions
        self.inputs = inputs
        self.targets = targets or []
        self.test_id = test_id
        self.input_ids = input_ids or []
        self._sec_level: SecLevel | None = None

    @property
    def sec_level(self) -> SecLevel | None:
        """Get the security level for this Component.

        Returns:
            SecLevel if present, None otherwise
        """
        return self._sec_level

    def parts(self) -> list[Component | CBlock]:
        """The set of constituent parts of the Component."""
        return []

    def format_for_llm(self) -> TemplateRepresentation:
        """Formats the test for judge evaluation."""
        return TemplateRepresentation(
            obj=self,
            args=self._judge_context if hasattr(self, "_judge_context") else {},
            template_order=["*"],
        )

    def _parse(self, computed: ModelOutputThunk) -> str:
        """Parse the model output. Returns string value for now."""
        return computed.value if computed.value is not None else ""

    def set_judge_context(
        self, input_text: str, prediction: str, targets_for_input: list[str]
    ):
        """Set context for judge evaluation."""
        if len(targets_for_input) == 0:  # no reference
            target_text = "N/A"
        elif len(targets_for_input) == 1:
            target_text = targets_for_input[0]
        else:  # enumerate when there are multiple targets
            target_text = "\n".join(
                [f"{i}. {target}" for i, target in enumerate(targets_for_input, 1)]
            )

        self._judge_context: dict[str, Any] = {
            "input": input_text,
            "prediction": prediction,
            "target": target_text,
            "guidelines": self.instructions,
        }

    @classmethod
    def from_json_file(cls, filepath: str) -> list["TestBasedEval"]:
        """Load test evaluations from json/jsonl file, return list of TestBasedEval instances, one per 'unit test'."""
        path = Path(filepath)

        with path.open("r") as f:
            data = json.load(f)

        if not isinstance(data, list):
            data = [data]

        test_evals = []
        for test_data_dict in data:
            try:
                test_data = TestData(**test_data_dict)
            except Exception as e:
                raise ValueError(f"Invalid test data in {filepath}: {e}")

            inputs = []
            targets = []
            input_ids = []

            for example in test_data.examples:
                user_messages = [msg for msg in example.input if msg.role == "user"]
                if user_messages:
                    inputs.append(user_messages[-1].content)

                targets_for_input = [
                    msg.content for msg in example.targets if msg.role == "assistant"
                ]
                targets.append(targets_for_input)

                input_ids.append(example.input_id)

            test_eval = cls(
                source=test_data.source,
                name=test_data.name,
                instructions=test_data.instructions,
                inputs=inputs,
                targets=targets,
                test_id=test_data.id,
                input_ids=input_ids,
            )
            test_evals.append(test_eval)

        return test_evals
