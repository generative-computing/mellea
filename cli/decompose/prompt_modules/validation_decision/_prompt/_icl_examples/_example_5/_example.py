from .._types import ICLExample

# Example 5: LLM validation case
# This example demonstrates a requirement that involves semantic content interpretation
# The requirement specifies structural elements that need to be present in test cases

requirement = """Each test case should include: Test Case description, Precondition, Test Steps, and Expected Outcome"""

reasoning = """This requirement involves checking if test cases contain specific semantic elements with proper structure. While it might seem like a simple checklist, validating this properly requires understanding the content and context of each element. An LLM is better suited to determine if the provided text actually represents meaningful test case components rather than just checking for keyword presence. For example, it needs to distinguish between a genuine "Precondition" description versus just the word "Precondition" appearing randomly in text. This semantic understanding and contextual interpretation makes it more appropriate for LLM validation."""

decision = "llm"

example: ICLExample = {
    "requirement": requirement,
    "reasoning": reasoning,
    "decision": decision,
}
