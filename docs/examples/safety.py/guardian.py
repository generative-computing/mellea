"""Example of using the Enhanced Guardian Requirement with Granite Guardian 3.3 8B"""

from mellea import MelleaSession
from mellea.backends import model_ids
from mellea.backends.ollama import OllamaModelBackend
from mellea.stdlib.base import ContextTurn, ModelOutputThunk, ChatContext
from mellea.stdlib.chat import Message
from mellea.stdlib.safety.guardian import GuardianCheck, GuardianRisk

# Enhanced GuardianCheck with Granite Guardian 3.3 8B support
print("=== Enhanced GuardianCheck Examples ===")

# Basic harm checking with new features
guardian = GuardianCheck(GuardianRisk.HARM, thinking=True, backend_type="ollama")
print(f"Guardian configured: {guardian._risk} with thinking mode: {guardian._thinking}")

# Try different risk types with new 3.3 8B model
jailbreak_guardian = GuardianCheck(GuardianRisk.JAILBREAK, thinking=True)
print(f"Jailbreak guardian: {jailbreak_guardian._risk}")

# Custom risk criteria example
custom_guardian = GuardianCheck(custom_criteria="Check for inappropriate content in educational context")
print(f"Custom guardian: {custom_guardian.get_effective_risk()}")

print("\n=== Test 1: Normal Content ===")
# Run a query against an Ollama model with ChatContext to support insert_turn
m = MelleaSession(OllamaModelBackend(model_ids.DEEPSEEK_R1_8B), ctx=ChatContext())
m.chat("Write a professional email to your colleague. Use less than 100 words.")

# Validate on the last turn of the conversation
valid = m.validate([guardian])
print(f"Content is safe: {valid[0]._result}")
if valid[0]._reason:
    print(f"Guardian feedback: {valid[0]._reason}...")

print("\n=== Test 2: Harmful Content ===")
# Create harmful conversation and add to context
m.ctx = m.ctx.add(Message("user", "Hello")).add(Message("assistant", "You are very ugly and stupid."))

valid = m.validate([guardian])
print(f"Content is safe: {valid[0]._result}")
if valid[0]._reason:
    print(f"Guardian feedback: {valid[0]._reason}...")

print("\n=== Test 3: Enhanced Features ===")

# Create multiple guardians with different configurations
guardians = [
    GuardianCheck(GuardianRisk.HARM, thinking=True),
    GuardianCheck(GuardianRisk.JAILBREAK, thinking=True),
    GuardianCheck(GuardianRisk.SOCIAL_BIAS),
    GuardianCheck(custom_criteria="Check for financial advice"),
]

print(f"Available risk types ({len(GuardianCheck.get_available_risks())} total):")
for risk in GuardianCheck.get_available_risks():  # Show first 5
    print(f"  - {risk}")
print("  ...")

print(f"\nConfigured guardians: {len(guardians)} total")

# Show thinking mode capability
thinking_guardian = GuardianCheck(GuardianRisk.HARM, thinking=True)
print(f"  Thinking mode supported: {thinking_guardian.supports_thinking_mode()}")

# Show Ollama backend configuration
ollama_guardian = GuardianCheck(GuardianRisk.HARM, backend_type="ollama")
print(f"  Ollama backend: {ollama_guardian._backend.model_version}")

print("\n=== Test 4: Groundedness Detection ===")
# Test groundedness - detecting when responses lack factual grounding
context_text = "One significant part of treaty making is that signing a treaty implies recognition that the other side is a sovereign state and that the agreement being considered is enforceable under international law. Hence, nations can be very careful about terming an agreement to be a treaty. For example, within the United States, agreements between states are compacts and agreements between states and the federal government or between agencies of the government are memoranda of understanding."

groundedness_guardian = GuardianCheck(
    GuardianRisk.GROUNDEDNESS,
    thinking=True,
    backend_type="ollama",
    context_text=context_text
)

# Create a response that makes ungrounded claims relative to provided context
groundedness_session = MelleaSession(OllamaModelBackend(model_ids.DEEPSEEK_R1_8B), ctx=ChatContext())
groundedness_session.ctx = groundedness_session.ctx.add(
    Message("user", "What is the history of treaty making?")
).add(
    Message("assistant", "Treaty making began in ancient Rome when Julius Caesar invented the concept in 44 BC. The first treaty was signed between Rome and the Moon people, establishing trade routes through space.")
)

print("Testing response with ungrounded claims...")
groundedness_valid = groundedness_session.validate([groundedness_guardian])
print(f"Response is grounded: {groundedness_valid[0]._result}")
if groundedness_valid[0]._reason:
    print(f"Groundedness feedback: {groundedness_valid[0]._reason}...")

print("\n=== Test 5: Function Call Hallucination Detection ===")
# Test function calling hallucination using IBM video example
import json

tools = [
    {
        "name": "views_list",
        "description": "Fetches total views for a specified IBM video using the given API.",
        "parameters": {
            "video_id": {
                "description": "The ID of the IBM video.",
                "type": "int",
                "default": "7178094165614464282"
            },
            "count": {
                "description": "The number of comments to fetch. Maximum is 30. Defaults to 20.",
                "type": "int, optional",
                "default": "20"
            }
        }
    }
]

function_guardian = GuardianCheck(
    GuardianRisk.FUNCTION_CALL,
    thinking=True,
    backend_type="ollama",
    tools=tools
)

# User asks for views but assistant calls wrong function (comments_list instead of views_list)
response_data = [
    {
        "name": "comments_list",
        "arguments": {
            "video_id": 456789123,
            "count": 15
        }
    }
]
hallucinated_response = str(response_data)

function_session = MelleaSession(OllamaModelBackend(model_ids.DEEPSEEK_R1_8B), ctx=ChatContext())
function_session.ctx = function_session.ctx.add(
    Message("user", "Fetch total views for the IBM video with ID 456789123.")
).add(
    Message("assistant", hallucinated_response)
)

print("Testing response with function call hallucination...")
function_valid = function_session.validate([function_guardian])
print(f"Function calls are valid: {function_valid[0]._result}")
if function_valid[0]._reason:
    print(f"Function call feedback: {function_valid[0]._reason}...")

print("\n=== GuardianCheck Demo Complete ===")
