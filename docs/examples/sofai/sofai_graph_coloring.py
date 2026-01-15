"""SOFAI Sampling Strategy Example: Graph Coloring Problem.

This example demonstrates the SOFAI (Slow and Fast AI)
sampling strategy using a graph coloring constraint satisfaction problem.

In this example, we use the SOFAI sampling strategy. Because we wrote this
example to run on consumer grade hardware, each model is still relatively small:
1. S1 Solver (phi:2.7b) - Fast model with iterative feedback loop
2. S2 Solver (qwen3-4b-thinking) - Slow model, called once as fallback
3. Custom validator - Provides detailed feedback for constraint violations

Note: This example uses a custom validator (check_graph_coloring). To use the
optional judge_backend feature for LLM-as-Judge validation, you can:
- Remove the validation_fn parameter from req()
- Add judge_backend and feedback_strategy parameters to SOFAISamplingStrategy
- feedback_strategy options: "simple", "first_error", "all_errors"
"""

import json
import logging

import mellea
from mellea.backends.ollama import OllamaModelBackend
from mellea.core import FancyLogger
from mellea.stdlib.components import Message
from mellea.stdlib.context import ChatContext
from mellea.stdlib.requirements import ValidationResult, req
from mellea.stdlib.sampling import SOFAISamplingStrategy

# Define the graph coloring problem
graph = {"A": ["B"], "B": ["A", "C"], "C": ["B"]}
colors = ["Red", "Blue"]

graph_description = (
    f"Color the nodes of the graph (A, B, C) using at most {len(colors)} colors "
    f"({', '.join(colors)}). Adjacent nodes must have different colors. "
    f"The adjacencies are: A is adjacent to B and C; B is adjacent to A and C; "
    f"C is adjacent to A and B."
)

output_format_instruction = (
    "Provide the solution as a JSON object where keys are node names "
    'and values are the assigned color strings (e.g., {"A": "Red", "B": "Green", ...}).'
)


def parse_coloring(output_str: str) -> dict | None:
    """Parse LLM output as JSON, handling markdown code blocks."""
    try:
        # Remove markdown code blocks if present
        output_str = output_str.strip()
        if output_str.startswith("```json"):
            output_str = output_str[7:].split("```")[0].strip()
        elif output_str.startswith("```"):
            output_str = output_str[3:].split("```")[0].strip()

        parsed = json.loads(output_str)
        if not isinstance(parsed, dict):
            return None
        return parsed
    except (json.JSONDecodeError, Exception):
        return None


def check_graph_coloring(ctx) -> ValidationResult:
    """Validate graph coloring with detailed, targeted feedback.

    This validator provides specific reasons for failures, which SOFAI
    uses to generate targeted repair messages for the LLM.
    """
    # Extract output from context
    output = ctx.last_output()
    if output is None:
        return ValidationResult(
            False, reason="No output found. " + output_format_instruction
        )

    # Parse the coloring
    coloring = parse_coloring(str(output.value))
    if coloring is None:
        return ValidationResult(
            False,
            reason=f"Could not parse output as valid JSON. Expected format: {output_format_instruction}",
        )

    # Collect all errors for detailed feedback
    errors = []

    # Check all nodes are colored
    colored_nodes = set(coloring.keys())
    graph_nodes = set(graph.keys())
    missing = graph_nodes - colored_nodes
    extra = colored_nodes - graph_nodes

    if missing:
        errors.append(f"Missing nodes: {', '.join(sorted(missing))}")
    if extra:
        errors.append(f"Unexpected nodes: {', '.join(sorted(extra))}")

    # Check valid colors
    invalid_colors = [c for c in coloring.values() if c not in colors]
    if invalid_colors:
        errors.append(
            f"Invalid colors {set(invalid_colors)}. Use only: {', '.join(colors)}"
        )

    # Check adjacency constraints (only if basic structure is valid)
    if not errors:
        for node, neighbors in graph.items():
            if node not in coloring:
                continue
            node_color = coloring[node]
            for neighbor in neighbors:
                if neighbor in coloring and coloring[neighbor] == node_color:
                    errors.append(
                        f"Nodes {node} and {neighbor} are adjacent but both have color '{node_color}'"
                    )

    if errors:
        return ValidationResult(False, reason=" | ".join(errors))
    else:
        return ValidationResult(True, reason="Valid graph coloring!")


# Define the requirement with custom validator
requirements = [
    req(
        description="The coloring must be valid (adjacent nodes have different colors, all nodes colored, correct JSON format).",
        validation_fn=check_graph_coloring,
    )
]


def main():
    """Run the graph coloring example with SOFAI strategy."""
    # Initialize backends
    s1_solver_backend = OllamaModelBackend(model_id="phi:2.7b")
    s2_solver_backend = OllamaModelBackend(
        model_id="pielee/qwen3-4b-thinking-2507_q8:latest"
    )

    # Optional: Initialize judge backend for LLM-as-Judge validation
    # Uncomment to use a third model for validation instead of custom validator
    # judge_backend = OllamaModelBackend(model_id="llama3.2:3b")

    # Create SOFAI strategy
    sofai_strategy = SOFAISamplingStrategy(
        s1_solver_backend=s1_solver_backend,
        s2_solver_backend=s2_solver_backend,
        s2_solver_mode="fresh_start",  # Options: "fresh_start", "continue_chat", "best_attempt"
        loop_budget=3,
        # judge_backend=judge_backend,  # Uncomment to use judge backend
        # feedback_strategy="all_errors",  # Options: "simple", "first_error", "all_errors"
    )

    # Create session with S1 solver as default backend
    # Note: SOFAI requires ChatContext for multi-turn conversation
    m = mellea.MelleaSession(backend=s1_solver_backend, ctx=ChatContext())

    print("--- Starting Graph Coloring with SOFAI Strategy ---")
    problem_prompt = f"{graph_description}\n{output_format_instruction}"

    # Run sampling with SOFAI strategy
    sampling_result = m.instruct(
        problem_prompt,
        requirements=requirements,
        strategy=sofai_strategy,
        return_sampling_results=True,
        model_options={"temperature": 0.1, "seed": 42},
    )

    print("\n--- SOFAI Strategy Results ---")
    print(f"Success: {sampling_result.success}")
    print(f"Number of attempts: {len(sampling_result.sample_generations)}")

    # Determine which solver produced each result
    solver_1_attempts = min(
        sofai_strategy.loop_budget, len(sampling_result.sample_generations)
    )

    for i, (action, gen, val_list) in enumerate(
        zip(
            sampling_result.sample_actions,
            sampling_result.sample_generations,
            sampling_result.sample_validations,
        )
    ):
        print(f"\n--- Attempt {i + 1} ---")

        # Determine which solver was used
        if i < solver_1_attempts:
            solver_name = "S1 Solver (phi:2.7b)"
        else:
            solver_name = "S2 Solver (qwen3-4b-thinking)"

        print(f"Solver: {solver_name}")

        # Show the action (original instruction or repair message)
        if isinstance(action, Message):
            print("Action: [Repair Message]")
            print(f"Content: {action.content[:150]}...")
        elif hasattr(action, "description"):
            print("Action: [Original Instruction]")
        else:
            print(f"Action: {type(action).__name__}")

        # Show generated output
        print(f"Output: {gen.value}")

        # Show validation results
        print("Validation:")
        passed_all = True
        for req_obj, val_result in val_list:
            status = "✓ PASS" if val_result.as_bool() else "✗ FAIL"
            print(f"  {status}: {req_obj.description}")
            if val_result.reason:
                print(f"    Reason: {val_result.reason}")
            if not val_result.as_bool():
                passed_all = False

        if passed_all:
            print(">> This attempt PASSED all requirements! <<")


if __name__ == "__main__":
    # Set logging level
    FancyLogger.get_logger().setLevel(logging.INFO)
    main()
