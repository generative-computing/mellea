from .._types import ICLExample

# Example 1: Code validation case
# This example demonstrates a requirement that can be validated with code
# The requirement is specific, measurable, and has clear success criteria

requirement = """Don't mention the word "water"."""

reasoning = """This requirement specifies that a certain word ("water") must not appear in the content. It can be validated deterministically by checking if the word "water" (case-insensitive) appears anywhere in the text. This is a straightforward string operation with clearly defined success/failure criteria - the validation passes if the word is not found and fails if it is found."""

decision = "code"

example: ICLExample = {
    "requirement": requirement,
    "reasoning": reasoning,
    "decision": decision,
}
