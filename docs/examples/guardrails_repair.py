# pytest: ollama, llm
"""Example demonstrating guardrail repair strategies.

This example shows how to use guardrails with repair enabled (check_only=False)
to guide the LLM in self-correcting outputs that fail validation.
"""

from mellea import start_session
from mellea.stdlib.requirements.guardrails import (
    contains_keywords,
    excludes_keywords,
    json_valid,
    max_length,
    min_length,
    no_pii,
)
from mellea.stdlib.sampling import RepairTemplateStrategy


def example_repair_pii():
    """Example: Repair output containing PII."""
    print("\n" + "=" * 80)
    print("EXAMPLE: Repair PII Detection")
    print("=" * 80)

    session = start_session()

    # With check_only=False, the guardrail provides actionable repair guidance
    result = session.instruct(
        "Generate a sample customer profile with name, email, and phone",
        requirements=[no_pii(check_only=False)],
        strategy=RepairTemplateStrategy(loop_budget=3),
    )

    print(f"\nFinal output (should be PII-free):\n{result.value}")


def example_repair_json():
    """Example: Repair invalid JSON output."""
    print("\n" + "=" * 80)
    print("EXAMPLE: Repair JSON Format")
    print("=" * 80)

    session = start_session()

    result = session.instruct(
        "Create a JSON object with fields: name (string), age (number), active (boolean)",
        requirements=[json_valid(check_only=False)],
        strategy=RepairTemplateStrategy(loop_budget=3),
    )

    print(f"\nFinal output (should be valid JSON):\n{result.value}")


def example_repair_length():
    """Example: Repair output that's too long or too short."""
    print("\n" + "=" * 80)
    print("EXAMPLE: Repair Length Constraints")
    print("=" * 80)

    session = start_session()

    # Too long - should be shortened
    result1 = session.instruct(
        "Write a brief summary of Python programming",
        requirements=[max_length(100, unit="characters", check_only=False)],
        strategy=RepairTemplateStrategy(loop_budget=3),
    )

    print(f"\nShortened output ({len(result1.value or '')} chars):\n{result1.value}")

    # Too short - should be expanded
    result2 = session.instruct(
        "Explain machine learning",
        requirements=[min_length(200, unit="characters", check_only=False)],
        strategy=RepairTemplateStrategy(loop_budget=3),
    )

    print(f"\nExpanded output ({len(result2.value or '')} chars):\n{result2.value}")


def example_repair_keywords():
    """Example: Repair missing or forbidden keywords."""
    print("\n" + "=" * 80)
    print("EXAMPLE: Repair Keyword Requirements")
    print("=" * 80)

    session = start_session()

    # Missing keywords - should add them
    result1 = session.instruct(
        "Explain web development",
        requirements=[
            contains_keywords(
                ["HTML", "CSS", "JavaScript"], require_all=True, check_only=False
            )
        ],
        strategy=RepairTemplateStrategy(loop_budget=3),
    )

    print(f"\nOutput with required keywords:\n{result1.value}")

    # Forbidden keywords - should remove them
    result2 = session.instruct(
        "Write professional documentation about the project",
        requirements=[excludes_keywords(["TODO", "FIXME", "hack"], check_only=False)],
        strategy=RepairTemplateStrategy(loop_budget=3),
    )

    print(f"\nProfessional output (no forbidden keywords):\n{result2.value}")


def example_combined_repair():
    """Example: Multiple guardrails with repair."""
    print("\n" + "=" * 80)
    print("EXAMPLE: Combined Repair Strategies")
    print("=" * 80)

    session = start_session()

    # Multiple constraints that may need repair
    result = session.instruct(
        "Generate a JSON user profile",
        requirements=[
            json_valid(check_only=False),
            no_pii(check_only=False),
            max_length(300, check_only=False),
            contains_keywords(["username", "role"], check_only=False),
        ],
        strategy=RepairTemplateStrategy(loop_budget=5),
    )

    print(f"\nFinal output (meets all requirements):\n{result.value}")


def example_check_only_vs_repair():
    """Example: Comparing check_only=True vs check_only=False."""
    print("\n" + "=" * 80)
    print("EXAMPLE: Check-Only vs Repair Mode")
    print("=" * 80)

    session = start_session()

    # check_only=True: Brief reason, hard fail
    print("\nWith check_only=True (validation only):")
    try:
        result1 = session.instruct(
            "Write a 500-word essay",
            requirements=[max_length(50, check_only=True)],
            strategy=RepairTemplateStrategy(loop_budget=1),
        )
        print(f"Output: {result1.value}")
    except Exception as e:
        print(f"Failed: {e}")

    # check_only=False: Detailed guidance, repair attempts
    print("\nWith check_only=False (repair enabled):")
    result2 = session.instruct(
        "Write a brief summary",
        requirements=[max_length(50, check_only=False)],
        strategy=RepairTemplateStrategy(loop_budget=3),
    )
    print(f"Output ({len(result2.value or '')} chars): {result2.value}")


if __name__ == "__main__":
    # Run examples
    example_repair_json()
    example_repair_length()
    example_repair_keywords()
    example_combined_repair()
    example_check_only_vs_repair()

    # Note: example_repair_pii() requires careful testing as it may
    # generate PII in the first attempt before repair

    print("\n" + "=" * 80)
    print("ALL REPAIR EXAMPLES COMPLETE")
    print("=" * 80)
