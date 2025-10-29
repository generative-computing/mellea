"""LLM Evaluation with Unit Tests in Mellea."""

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mellea.stdlib.base import CBlock, Component, TemplateRepresentation


class TestBasedEval(Component):
    """Each TestBasedEval represents a single unit test."""

    def __init__(
        self,
        category: str,
        judge_prompt: str,
        inputs: list[str],
        unit_test_instructions: str,
        targets: list[str] | None = None,  # can be optional
        conversation_id: str | None = None,  # in case we change format later
        metadata: dict[str, Any] | None = None,  # for anything miscellaneous
    ):
        """Initialize TestBasedEval (for a single unit test)."""
        self.category = category
        self.judge_prompt = judge_prompt
        self.inputs = inputs
        self.unit_test_instructions = unit_test_instructions
        self.targets = targets or []
        self.conversation_id = conversation_id
        self.metadata = metadata or {}

    def format_for_llm(self) -> TemplateRepresentation | str:
        """Format for use by judge LLM session."""
        return self.category + "\n\n" + self.judge_prompt

        # if making a new jinja template
        # return TemplateRepresentation(
        #     obj=self,
        #     args={
        #         "category": self.category,
        #         "judge_prompt": self.judge_prompt,
        #         "inputs": self.inputs,
        #         "unit_test_instructions": self.unit_test_instructions,
        #         "targets": self.targets,
        #     },
        #     #template_order=["*"]
        # )

    @classmethod
    def from_json_file(cls, filepath: str) -> list["TestBasedEval"]:
        """Load test evaluations from json/jsonl file, return list of TestBasedEval instances, one per 'unit test'."""
        path = Path(filepath)
        test_data = []

        if path.suffix == ".jsonl":
            with path.open("r") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        if not data.get("skip", False):
                            test_data.append(data)
        else:  # '.json'
            with path.open("r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    test_data = [item for item in data if not item.get("skip", False)]

        # group by conversation_id -- a single 'unit test'
        grouped = defaultdict(list)
        for item in test_data:
            conversation_id = item.get("conversation_id")
            grouped[conversation_id].append(item)

        test_evals = []
        for conversation_id, items in grouped.items():
            first_item = items[0]
            ut_instructions = first_item.get("unit test instructions", "")
            category = first_item.get("dataset", path.stem)

            inputs = []
            for item in items:
                for turn in item.get("input", []):
                    if turn.get("speaker") == "user":
                        inputs.append(turn.get("text"))

            targets = []
            for item in items:
                for turn in item.get("targets", []):
                    if turn.get("speaker") == "agent":
                        targets.append(turn.get("text"))

            # figure out how to add judge prompt template here
            judge_prompt = """**Input to the model**

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

            metadata = {
                "task_type": first_item.get(
                    "task_type"
                ),  # for displaying results maybe?
                "dataset": first_item.get("dataset"),
                "items": items,  # keep list of original items for reference
            }

            test_eval = cls(
                category=category,
                judge_prompt=judge_prompt,
                inputs=inputs,
                unit_test_instructions=ut_instructions,
                targets=targets,
                conversation_id=conversation_id,
                metadata=metadata,
            )
            test_evals.append(test_eval)

        return test_evals
