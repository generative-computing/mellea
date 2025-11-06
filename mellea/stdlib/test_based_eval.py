"""LLM Evaluation with Unit Tests in Mellea."""

import json
from pathlib import Path
from typing import Any

from mellea.stdlib.base import Component


class TestBasedEval(Component):
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

        self.judge_prompt = """**Input to the model**

            {input}

            **Model output to be rated**

            {prediction}

            **Ground truth text**

            {target}

            **Rating Guidelines**
            The model output should adhere to the following guidelines:
             {guidelines}

            **Scoring Criteria**
             * Score 0: The model output violates any of the guidelines.
             * Score 1: The model output is well aligned with the ground truth - if it exists, the input to the model, and adheres to all guidelines.

            **Return Your Rating**
               Return your rating in the following format:
               {{\"score\": your_score, \"justification\": \"your_justification\"}}

            Your rating:
            """

    @classmethod
    def from_json_file(cls, filepath: str) -> list["TestBasedEval"]:
        """Load test evaluations from json/jsonl file, return list of TestBasedEval instances, one per 'unit test'."""
        path = Path(filepath)

        with path.open("r") as f:
            data = json.load(f)

        if not isinstance(data, list):
            data = [data]

        test_evals = []
        for test_data in data:
            examples = test_data.get("examples", [])

            inputs = []
            targets = []
            input_ids = []

            for example in examples:
                input_messages = example.get("input", [])
                user_messages = [
                    msg for msg in input_messages if msg.get("role") == "user"
                ]
                if user_messages:
                    inputs.append(user_messages[-1].get("content", ""))

                target_messages = example.get("targets", [])
                targets_for_input = [
                    msg.get("content", "")
                    for msg in target_messages
                    if msg.get("role") == "assistant"
                ]
                targets.append(targets_for_input)

                input_ids.append(example.get("input_id", ""))

            test_eval = cls(
                source=test_data.get("source", "unknown"),
                name=test_data.get("name", ""),
                instructions=test_data.get("instructions", ""),
                inputs=inputs,
                targets=targets,
                test_id=test_data.get("id", ""),
                input_ids=input_ids,
            )
            test_evals.append(test_eval)

        return test_evals
