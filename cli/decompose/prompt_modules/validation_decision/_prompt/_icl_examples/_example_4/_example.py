from .._types import ICLExample

# Example 4: LLM validation case
# This example demonstrates a requirement that involves creative evaluation

requirement = """The generated marketing copy should be compelling and persuasive, effectively communicating the product's value proposition to potential customers."""

reasoning = """This requirement involves evaluating the quality of creative content (marketing copy) based on subjective criteria like 'compelling', 'persuasive', and 'effectively communicating'. These qualities require nuanced judgment and contextual understanding that cannot be easily codified into deterministic algorithms. This is best evaluated by an LLM with human-like comprehension."""

decision = "llm"

example: ICLExample = {
    "requirement": requirement,
    "reasoning": reasoning,
    "decision": decision,
}
