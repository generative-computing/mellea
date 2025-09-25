"""Example of using GuardianCheck with HuggingFace backend for direct model inference"""

from mellea.stdlib.base import ContextTurn, ModelOutputThunk
from mellea.stdlib.chat import Message
from mellea.stdlib.safety.guardian import GuardianCheck, GuardianRisk

print("=== GuardianCheck HuggingFace Backend Example ===")

# Test harmful content detection using HuggingFace backend
print("Testing harm detection with HuggingFace backend...")
hf_guardian = GuardianCheck(GuardianRisk.HARM, thinking=False, backend_type="huggingface")

print(f"Backend: {hf_guardian._backend.model_version}")
print(f"Thinking mode: {hf_guardian._thinking}")

# Test the "ugly and stupid" example directly with Guardian
print("\n=== Direct Guardian Validation Test ===")
print("Testing content: 'You are very ugly and stupid.'")

try:
    # Create a minimal context for validation
    from mellea.stdlib.base import LinearContext

    # Create context with the harmful content
    ctx = LinearContext()
    ctx.insert_turn(
        ContextTurn(
            Message("user", "Hello"),
            ModelOutputThunk("You are very ugly and stupid.")
        )
    )

    # Run Guardian validation directly
    validation_result = hf_guardian._guardian_validate(ctx)

    print(f"Content is safe: {validation_result.as_bool()}")
    print(f"Guardian detected harm: {not validation_result.as_bool()}")

    if validation_result.reason:
        print(f"\nGuardian reasoning:")
        print(validation_result.reason)

except Exception as e:
    print(f"HuggingFace backend test failed: {e}")

print("\n=== HuggingFace Guardian Demo Complete ===")