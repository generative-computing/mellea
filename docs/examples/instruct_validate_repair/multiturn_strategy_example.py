# pytest: ollama, llm, qualitative

"""MultiTurnStrategy Example with Validation Functions

Demonstrates how MultiTurnStrategy uses conversational repair with detailed
validation feedback to iteratively improve outputs.

This example shows the key difference between MultiTurnStrategy and other strategies:
it builds a conversation history where validation failures are communicated as user
messages, allowing the model to iteratively improve its response through dialogue.
"""

from mellea import start_session
from mellea.backends import ModelOption
from mellea.stdlib.context import ChatContext
from mellea.stdlib.requirements import req
from mellea.stdlib.requirements.requirement import simple_validate
from mellea.stdlib.sampling import MultiTurnStrategy

MIN_WORD_COUNT = 100


def validate_word_count(text: str) -> tuple[bool, str]:
    """A validation function that checks for minimum word count.

    Returns detailed failure reasons to help the model understand what's wrong.
    """
    word_count = len(text.split())
    if word_count < MIN_WORD_COUNT:
        return (
            False,
            f"Output has only {word_count} words. Need at least {MIN_WORD_COUNT} words.",
        )
    return True, ""


def demo_multiturn_repair():
    """Demonstrate MultiTurnStrategy with detailed validation feedback."""

    # MultiTurnStrategy requires ChatContext for conversational repair
    m = start_session(
        ctx=ChatContext(), model_options={ModelOption.MAX_NEW_TOKENS: 300}
    )

    print("=== MultiTurnStrategy Demo ===\n")
    print("Task: Write a detailed explanation of quantum computing\n")

    result = m.instruct(
        "Explain quantum computing like I am 5.",
        requirements=[
            req(
                "Must be at least 100 words",
                validation_fn=simple_validate(validate_word_count),
            ),
            "Include at least one real-world application",
            "Avoid technical jargon",
        ],
        strategy=MultiTurnStrategy(loop_budget=5),
        return_sampling_results=True,
    )

    # Show the repair process
    print(f"Attempts made: {len(result.sample_generations)}")
    print(f"Success: {result.success}\n")

    for i, (gen, validations) in enumerate(
        zip(result.sample_generations, result.sample_validations), 1
    ):
        print(f"\n--- Attempt {i} ---")
        output_text = str(gen.value) if gen.value else ""
        print(f"Output length: {len(output_text.split())} words")

        failed = [v for _, v in validations if not v.as_bool()]
        if failed:
            print("Failed validations:")
            for val in failed:
                if val.reason:
                    print(f"  - {val.reason}")
        else:
            print("✓ All validations passed!")

    print(f"\n{'=' * 60}")
    print("Final output:")
    print(f"{'=' * 60}")
    print(result.value)
    print(f"{'=' * 60}")

    # Show the conversation history
    print("\nConversation history:")
    for i, msg in enumerate(m.ctx.as_list(), 1):
        role = getattr(msg, "role", "unknown")
        content = str(getattr(msg, "content", msg))[:100]
        print(f"{i}. [{role}] {content}...")

    return result


if __name__ == "__main__":
    demo_multiturn_repair()
