"""Example of using GuardianCheck with HuggingFace backend for direct model inference

This example shows how to reuse the Guardian backend across multiple validators
to avoid reloading the model multiple times.
"""

from mellea import MelleaSession
from mellea.backends import model_ids
from mellea.backends.huggingface import LocalHFBackend
from mellea.backends.ollama import OllamaModelBackend
from mellea.stdlib.base import ChatContext, ModelOutputThunk, ModelToolCall
from mellea.stdlib.chat import Message
from mellea.stdlib.safety.guardian import GuardianCheck, GuardianRisk

print("=== GuardianCheck HuggingFace Backend Example ===")

# Create a single reusable session with Ollama backend for test content generation
# We'll reuse this session and just reset its context for each test
session = MelleaSession(OllamaModelBackend(model_ids.DEEPSEEK_R1_8B), ctx=ChatContext())

# Create a single shared HuggingFace backend for Guardian (loads model once)
print("Loading Granite Guardian model (this happens only once)...")
shared_guardian_backend = LocalHFBackend(model_id="ibm-granite/granite-guardian-3.3-8b")
print(f"Loaded backend: {shared_guardian_backend.model_id}\n")

# Create Guardian validators that share the backend (no model reloading!)
print("Creating harm guardian...")
harm_guardian = GuardianCheck(
    GuardianRisk.HARM, thinking=True, backend=shared_guardian_backend
)

# Test 1: Harmful content detection
print("\n=== Test 1: Harmful Content Detection ===")
print("Testing content: 'You are very ugly and stupid.'")

# Reset context for this test
session.ctx = (
    ChatContext()
    .add(Message("user", "Hello"))
    .add(Message("assistant", "You are very ugly and stupid."))
)

validation_result = session.validate([harm_guardian])
print(f"Content is safe: {validation_result[0]._result}")
print(f"Guardian detected harm: {not validation_result[0]._result}")

if validation_result[0]._reason:
    print("\nGuardian feedback:")
    print(validation_result[0]._reason[:200] + "...")

# Test 2: Groundedness detection
print("\n=== Test 2: Groundedness Detection ===")
context_text = (
    "Python is a high-level programming language created by Guido van Rossum in 1991."
)

# Create groundedness guardian with context (reuse shared backend)
print("Creating groundedness guardian...")
groundedness_guardian = GuardianCheck(
    GuardianRisk.GROUNDEDNESS,
    thinking=False,
    context_text=context_text,
    backend=shared_guardian_backend,
)

# Reset context with ungrounded response
session.ctx = (
    ChatContext()
    .add(Message("user", "Who created Python?"))
    .add(
        Message(
            "assistant",
            "Python was created by Dennis Ritchie in 1972 for use in Unix systems.",
        )
    )
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
        "parameters": {"location": {"description": "City name", "type": "string"}},
    }
]

# Create function call guardian (reuse shared backend)
print("Creating function call guardian...")
function_guardian = GuardianCheck(
    GuardianRisk.FUNCTION_CALL,
    thinking=False,
    tools=tools,
    backend=shared_guardian_backend,
)


# User asks for weather but model calls wrong function
def dummy_func(**kwargs):
    pass


hallucinated_tool_calls = {
    "get_stock_price": ModelToolCall(
        name="get_stock_price", func=dummy_func, args={"symbol": "AAPL"}
    )
}

hallucinated_output = ModelOutputThunk(
    value="Let me get the weather for you.", tool_calls=hallucinated_tool_calls
)

# Reset context with hallucinated function call
session.ctx = (
    ChatContext()
    .add(Message("user", "What's the weather in Boston?"))
    .add(hallucinated_output)
)

function_valid = session.validate([function_guardian])
print(f"Function calls are valid: {function_valid[0]._result}")
if function_valid[0]._reason:
    print(f"Function call feedback: {function_valid[0]._reason[:200]}...")

print("\n=== HuggingFace Guardian Demo Complete ===")
