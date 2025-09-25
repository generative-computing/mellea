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
    print("RepairTemplateStrategy with Actual Function Call Demo")
    print("-" * 52)

    # Use Llama3.2 which supports function calling
    m = MelleaSession(OllamaModelBackend("llama3.2"))

    # Define actual callable functions
    def get_weather(location: str) -> str:
        """Gets current weather information for a location"""
        return f"The current weather in {location} is sunny, 22°C with light winds."

    def get_recipe(dish_name: str) -> str:
        """Gets a cooking recipe for the specified dish"""
        return f"Recipe for {dish_name}: Cook ingredients together until done."

    def get_stock_price(symbol: str) -> str:
        """Gets current stock price for a given symbol. Symbol must be a valid stock ticker (3-5 uppercase letters)."""
        return f"Current stock price for {symbol} is $150.25"

    # All available tools - both model and Guardian use the same set
    all_tools = [
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

    # Function call validation using GuardianRisk.FUNCTION_CALL
    safety_requirements = [
        GuardianCheck(
            GuardianRisk.FUNCTION_CALL,
            thinking=True,
            tools=all_tools  # Guardian and model use same tools
        )
    ]

    print(f"Risk Type: {safety_requirements[0].get_effective_risk()}")
    print(f"Available Tools: {[tool['name'] for tool in all_tools]}")

    # Query that should trigger invalid stock symbol usage
    test_prompt = "What's the price of Tesla stock?"
    print(f"Main Model Prompt: {test_prompt}")

    # Model functions
    all_functions = [get_weather, get_recipe, get_stock_price]
    print(f"Model Available Functions: {[f.__name__ for f in all_functions]}")

    try:
        result = m.instruct(
            test_prompt,
            requirements=safety_requirements,
            strategy=RepairTemplateStrategy(loop_budget=3),
            return_sampling_results=True,
            model_options={
                "temperature": 0.7,  # Some randomness
                "seed": 789,
                "tools": all_functions,
                "system": "When users ask about stock prices, always use the full company name as the symbol parameter instead of the ticker symbol. For example, use 'Tesla Motors' instead of 'TSLA', 'Apple Inc' instead of 'AAPL', etc."
            },
            tool_calls=True
        )

        # Show repair process
        if hasattr(result, 'sample_validations') and result.sample_validations:
            for attempt_num, (generation, validations) in enumerate(zip(result.sample_generations, result.sample_validations), 1):
                print(f"\nAttempt {attempt_num}:")

                # Show model response (may be empty for function calls only)
                response = str(generation.value) if generation.value else "[Function calls only]"
                print(f"Model Response: {response}")

                # Show function calls made
                if hasattr(generation, 'tool_calls') and generation.tool_calls:
                    print("Function Calls Made:")
                    for name, tool_call in generation.tool_calls.items():
                        print(f"  - {name}({tool_call.args})")

                # Show validation results
                for req_item, validation in validations:
                    status = "PASSED" if validation.as_bool() else "FAILED"
                    print(f"Status: {status}")
                    if validation.reason:
                        print(f"Guardian Reason: {validation.reason}")

        print(f"\nFinal Result: {'SUCCESS' if result.success else 'FAILED'}")
        print(f"Attempts used: {len(result.sample_generations) if hasattr(result, 'sample_generations') else 1}")

        return result

    except Exception as e:
        print(f"Function calling failed: {e}")
        print("This may be because the model doesn't support function calling or Ollama is not running.")
        return None


def main():
    """Run RepairTemplateStrategy demo with actual function call validation."""
    try:
        print("=== Actual Function Calling with Guardian Validation Demo ===")
        result = demo_repair_with_actual_function_calling()

        if result is None:
            print("\nDemo failed. Please ensure:")
            print("1. Ollama is running")
            print("2. llama3.2 model is available")
            print("3. Model supports function calling")

        print("\nDemo completed.")
    except Exception as e:
        print(f"Error: {e}")
        print("Ensure Ollama is running with a model that supports function calling.")


if __name__ == "__main__":
    main()