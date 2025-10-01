"""
RepairTemplateStrategy Example with Actual Function Call Validation
Demonstrates how RepairTemplateStrategy repairs responses using actual function calls.
"""

from mellea import MelleaSession
from mellea.backends.ollama import OllamaModelBackend
from mellea.stdlib.safety.guardian import GuardianCheck, GuardianRisk
from mellea.stdlib.sampling import RepairTemplateStrategy


def demo_repair_with_actual_function_calling():
    """Demonstrate RepairTemplateStrategy with actual function calling and Guardian validation."""
    print("=== Guardian Repair Demo ===\n")

    # Use Llama3.2 which supports function calling
    m = MelleaSession(OllamaModelBackend("llama3.2"))

    # Define callable functions for the model
    def get_weather(location: str) -> str:
        """Gets current weather information for a location"""
        return f"Weather in {location}: sunny, 22°C"

    def get_recipe(dish_name: str) -> str:
        """Gets a cooking recipe for the specified dish"""
        return f"Recipe for {dish_name}: [recipe details]"

    def get_stock_price(symbol: str) -> str:
        """Gets current stock price for a given symbol. Symbol must be a valid stock ticker (3-5 uppercase letters)."""
        return f"Stock price for {symbol}: $150.25"

    # Tool schemas - Guardian validates against these
    tool_schemas = [
        {
            "name": "get_weather",
            "description": "Gets current weather information for a location",
            "parameters": {
                "location": {
                    "description": "The location to get weather for",
                    "type": "string"
                }
            }
        },
        {
            "name": "get_recipe",
            "description": "Gets a cooking recipe for the specified dish",
            "parameters": {
                "dish_name": {
                    "description": "The name of the dish to get a recipe for",
                    "type": "string"
                }
            }
        },
        {
            "name": "get_stock_price",
            "description": "Gets current stock price for a given symbol. Symbol must be a valid stock ticker (3-5 uppercase letters).",
            "parameters": {
                "symbol": {
                    "description": "The stock symbol to get price for (must be 3-5 uppercase letters)",
                    "type": "string"
                }
            }
        }
    ]

    # Guardian validates function calls against tool schemas
    guardian = GuardianCheck(
        GuardianRisk.FUNCTION_CALL,
        thinking=True,
        tools=tool_schemas
    )

    # Query that should trigger invalid stock symbol usage
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
            "tools": [get_weather, get_recipe, get_stock_price],
            "system": "When users ask about stock prices, always use the full company name as the symbol parameter instead of the ticker symbol. For example, use 'Tesla Motors' instead of 'TSLA'."
        },
        tool_calls=True
    )

    # Show repair process
    for attempt_num, (generation, validations) in enumerate(zip(result.sample_generations, result.sample_validations), 1):
        print(f"Attempt {attempt_num}:")

        # Show function calls made
        if hasattr(generation, 'tool_calls') and generation.tool_calls:
            for name, tool_call in generation.tool_calls.items():
                print(f"  Function: {name}({tool_call.args})")

        # Show validation results
        for req_item, validation in validations:
            status = "PASS" if validation.as_bool() else "FAIL"
            print(f"  {status}")

            # For failures, show repair feedback
            if not validation.as_bool() and validation.reason and attempt_num < len(result.sample_generations):
                print(f"  Repair: {validation.reason.split('Rationale:')[1].split('Response_error_span')[0].strip() if 'Rationale:' in validation.reason else validation.reason}")
        print()

    print(f"Result: {'SUCCESS' if result.success else 'FAILED'} after {len(result.sample_generations)} attempt(s)")
    return result


if __name__ == "__main__":
    demo_repair_with_actual_function_calling()