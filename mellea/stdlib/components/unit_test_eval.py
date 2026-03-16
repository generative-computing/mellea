"""LLM Evaluation with Unit Tests in Mellea."""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from ...core import CBlock, Component, ModelOutputThunk, TemplateRepresentation


class Message(BaseModel):
    """Schema for a message in the test data.

    Attributes:
        role (str): The role of the message sender (e.g. ``"user"`` or
            ``"assistant"``).
        content (str): The text content of the message.
    """

    role: str
    content: str


class Example(BaseModel):
    """Schema for an example in the test data.

    Attributes:
        input (list[Message]): The input messages for this example.
        targets (list[Message]): The expected target messages for scoring.
        input_id (str): An optional identifier for this input example.
    """

    input: list[Message]
    targets: list[Message] = Field(default_factory=list)
    input_id: str = ""


class TestData(BaseModel):
    """Schema for test data loaded from json.

    Attributes:
        source (str): Origin identifier for this test dataset.
        name (str): Human-readable name for this test dataset.
        instructions (str): Evaluation guidelines used by the judge model.
        examples (list[Example]): The individual input/target example pairs.
        id (str): Unique identifier for this test dataset.
    """

    source: str
    name: str
    instructions: str
    examples: list[Example] = Field(default_factory=list)
    id: str

    @field_validator("examples")
    @classmethod
    def validate_examples(cls, v: list[Example]) -> list[Example]:
        """Validate that the examples list is not empty.

        Args:
            v (list[Example]): The value of the ``examples`` field being
                validated.

        Returns:
            list[Example]: The validated examples list, unchanged.

        Raises:
            ValueError: If the examples list is empty.
        """
        if not v:
            raise ValueError("examples list cannot be empty")
        return v


class TestBasedEval(Component[str]):
    """Each TestBasedEval represents a single unit test.

    Args:
        source (str): Origin identifier for this test dataset.
        name (str): Human-readable name for this test.
        instructions (str): Evaluation guidelines used by the judge model.
        inputs (list[str]): The input texts for each example.
        targets (list[list[str]] | None): Expected target strings for each
            input. ``None`` is treated as an empty list.
        test_id (str | None): Optional unique identifier for this test.
        input_ids (list[str] | None): Optional identifiers for each input.

    """

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
        """Initialize TestBasedEval with source, name, instructions, inputs, and optional targets."""
        self.source = source
        self.name = name
        self.instructions = instructions
        self.inputs = inputs
        self.targets = targets or []
        self.test_id = test_id
        self.input_ids = input_ids or []

    def parts(self) -> list[Component | CBlock]:
        """Return the constituent parts of this component.

        Returns:
            list[Component | CBlock]: Always an empty list; the component
            renders entirely via ``format_for_llm``.
        """
        return []

    def format_for_llm(self) -> TemplateRepresentation:
        """Format this test for judge evaluation.

        Returns:
            TemplateRepresentation: A template representation containing the
            judge context (input, prediction, target, guidelines) set by
            ``set_judge_context``, or an empty args dict if no context has
            been set yet.
        """
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
    ) -> None:
        """Set the context dictionary used when formatting this test for judge evaluation.

        Args:
            input_text (str): The original input text shown to the model.
            prediction (str): The model's generated output to evaluate.
            targets_for_input (list[str]): Reference target strings for this
                input. An empty list results in ``"N/A"`` as the target text.
        """
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
        """Load test evaluations from a JSON file, returning one ``TestBasedEval`` per unit test.

        Args:
            filepath (str): Path to a JSON file containing one test-data object
                or a JSON array of test-data objects.

        Returns:
            list[TestBasedEval]: A list of ``TestBasedEval`` instances, one for
            each object found in the file.

        Raises:
            ValueError: If any test-data object in the file does not conform to
                the ``TestData`` schema.
        """
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
