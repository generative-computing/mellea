"""Example of using GuardianCheck with HuggingFace backend for direct model inference

NOTE: This example reuses a single session to avoid loading multiple models into memory.
We only need to load one model (for generating test content), and the Guardian validators
share their backend instances to minimize memory usage.
"""

from mellea import MelleaSession
from mellea.backends import model_ids
from mellea.backends.ollama import OllamaModelBackend
from mellea.stdlib.base import ChatContext, ModelOutputThunk, ModelToolCall
from mellea.stdlib.chat import Message
from mellea.stdlib.safety.guardian import GuardianCheck, GuardianRisk

print("=== GuardianCheck HuggingFace Backend Example ===")
print("NOTE: Loading Granite Guardian model (this may take a while on first run)...\n")

# Create a single reusable session with Ollama backend for test content generation
# We'll reuse this session and just reset its context for each test
session = MelleaSession(OllamaModelBackend(model_ids.DEEPSEEK_R1_8B), ctx=ChatContext())

# Create Guardian validators (these share the same HF backend internally via __deepcopy__)
print("Initializing Guardian validators...")
harm_guardian = GuardianCheck(GuardianRisk.HARM, thinking=True, backend_type="huggingface")
print(f"Backend: {harm_guardian._backend.model_version}")

# Test 1: Harmful content detection
print("\n=== Test 1: Harmful Content Detection ===")
print("Testing content: 'You are very ugly and stupid.'")

# Reset context for this test
session.ctx = ChatContext().add(
    Message("user", "Hello")
).add(
    Message("assistant", "You are very ugly and stupid.")
)

validation_result = session.validate([harm_guardian])
print(f"Content is safe: {validation_result[0]._result}")
print(f"Guardian detected harm: {not validation_result[0]._result}")

if validation_result[0]._reason:
    print(f"\nGuardian feedback:")
    print(validation_result[0]._reason[:200] + "...")

# Test 2: Groundedness detection
print("\n=== Test 2: Groundedness Detection ===")
context_text = "Python is a high-level programming language created by Guido van Rossum in 1991."

# Create groundedness guardian with context
groundedness_guardian = GuardianCheck(
    GuardianRisk.GROUNDEDNESS,
    thinking=False,
    backend_type="huggingface",
    context_text=context_text
)

# Reset context with ungrounded response
session.ctx = ChatContext().add(
    Message("user", "Who created Python?")
).add(
    Message("assistant", "Python was created by Dennis Ritchie in 1972 for use in Unix systems.")
)

groundedness_valid = session.validate([groundedness_guardian])
print(f"Response is grounded: {groundedness_valid[0]._result}")
if groundedness_valid[0]._reason:
    print(f"Groundedness feedback: {groundedness_valid[0]._reason[:200]}...")

# Test 3: Function call validation
print("\n=== Test 3: Function Call Validation ===")

tools = [
    {
        "name": "get_weather",
        "description": "Gets weather for a location",
        "parameters": {
            "location": {
                "description": "City name",
                "type": "string"
            }
        }
    }
]

function_guardian = GuardianCheck(
    GuardianRisk.FUNCTION_CALL,
    thinking=False,
    backend_type="huggingface",
    tools=tools
)

# User asks for weather but model calls wrong function
def dummy_func(**kwargs):
    pass

hallucinated_tool_calls = {
    "get_stock_price": ModelToolCall(
        name="get_stock_price",
        func=dummy_func,
        args={"symbol": "AAPL"}
    )
}

hallucinated_output = ModelOutputThunk(
    value="Let me get the weather for you.",
    tool_calls=hallucinated_tool_calls
)

# Reset context with hallucinated function call
session.ctx = ChatContext().add(
    Message("user", "What's the weather in Boston?")
).add(hallucinated_output)

function_valid = session.validate([function_guardian])
print(f"Function calls are valid: {function_valid[0]._result}")
if function_valid[0]._reason:
    print(f"Function call feedback: {function_valid[0]._reason[:200]}...")

print("\n=== HuggingFace Guardian Demo Complete ===")