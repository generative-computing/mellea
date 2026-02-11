# pytest: ollama, huggingface, requires_heavy_ram, llm

"""RepairTemplateStrategy Example with Actual Function Call Validation
Demonstrates how RepairTemplateStrategy repairs responses using actual function calls.
"""

from mellea import MelleaSession
from mellea.backends.ollama import OllamaModelBackend
from mellea.stdlib.requirements.safety.guardian import GuardianCheck, GuardianRisk
from mellea.stdlib.sampling import RepairTemplateStrategy

from mellea.backends.tools import MelleaTool


def demo_repair_with_actual_function_calling():
    """Demonstrate RepairTemplateStrategy with actual function calling and Guardian validation.

    Note: This demo uses an intentionally misconfigured system prompt to force an initial error,
    demonstrating how Guardian provides detailed repair feedback that helps the model correct itself.
    """
    print("=== Guardian Repair Demo ===\n")

    # Use Llama3.2 which supports function calling
    m = MelleaSession(OllamaModelBackend("llama3.2"))

    # Simple function for stock price
    def get_stock_price(symbol: str) -> str:
        """Gets current stock price for a given symbol. Symbol must be a valid stock ticker (3-5 uppercase letters)."""
        return f"Stock price for {symbol}: $150.25"

    # Tool schema - Guardian validates against this
    tool_schemas = [
        {
            "name": "get_stock_price",
            "description": "Gets current stock price for a given symbol. Symbol must be a valid stock ticker (3-5 uppercase letters).",
            "parameters": {
                "symbol": {
                    "description": "The stock symbol to get price for (must be 3-5 uppercase letters like TSLA, AAPL)",
                    "type": "string",
                }
            },
        }
    ]

    # Guardian validates function calls against tool schema
    guardian = GuardianCheck(
        GuardianRisk.FUNCTION_CALL, thinking=True, tools=tool_schemas
    )

    test_prompt = "What's the price of Tesla stock?"
    print(f"Prompt: {test_prompt}\n")

    result = m.instruct(
        test_prompt,
        requirements=[guardian],
        strategy=RepairTemplateStrategy(loop_budget=3),
        return_sampling_results=True,
        model_options={
            "temperature": 0.7,
            "seed": 789,
            "tools": [MelleaTool.from_callable(get_stock_price)],
            # Intentionally misconfigured to demonstrate repair
            "system": "When users ask about stock prices, use the full company name as the symbol parameter. For example, use 'Tesla Motors' instead of 'TSLA'.",
        },
        tool_calls=True,
    )

    # Show repair process
    for attempt_num, (generation, validations) in enumerate(
        zip(result.sample_generations, result.sample_validations), 1
    ):
        print(f"\nAttempt {attempt_num}:")

        # Show what was sent to the model
        if (
            hasattr(result, "sample_actions")
            and result.sample_actions
            and attempt_num <= len(result.sample_actions)
        ):
            action = result.sample_actions[attempt_num - 1]
            if hasattr(m.backend, "formatter"):
                try:
                    rendered = m.backend.formatter.print(action)
                    print("  Instruction sent to model:")
                    print("  ---")
                    print(f"  {rendered}")
                    print("  ---")
                except Exception:
                    pass

        # Show function calls made
        if hasattr(generation, "tool_calls") and generation.tool_calls:
            for name, tool_call in generation.tool_calls.items():
                print(f"  Function: {name}({tool_call.args})")

        # Show validation results
        for req_item, validation in validations:
            status = "PASS" if validation.as_bool() else "FAIL"
            print(f"  Status: {status}")

    print(f"\n{'=' * 60}")
    print(
        f"Result: {'SUCCESS' if result.success else 'FAILED'} after {len(result.sample_generations)} attempt(s)"
    )
    print(f"{'=' * 60}")
    return result


if __name__ == "__main__":
    demo_repair_with_actual_function_calling()
