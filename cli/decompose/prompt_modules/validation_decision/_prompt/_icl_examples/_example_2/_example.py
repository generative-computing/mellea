from .._types import ICLExample

# Example 2: LLM validation case
# This example demonstrates a requirement that needs subjective evaluation
# The requirement is qualitative and requires human-like judgment

requirement = """The user interface should be intuitive and provide a seamless experience for first-time users."""

reasoning = """This requirement is subjective and qualitative, focusing on user experience aspects like 'intuitive' and 'seamless'. These concepts cannot be measured with deterministic algorithms but require human-like judgment to evaluate. The assessment would depend on contextual understanding and interpretation of what constitutes a good user experience."""

decision = "llm"

example: ICLExample = {
    "requirement": requirement,
    "reasoning": reasoning,
    "decision": decision,
}
