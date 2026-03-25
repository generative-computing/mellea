from .._types import ICLExample

# Example 4: LLM validation case
# This example demonstrates a requirement that involves creative evaluation.

constraint_requirement = """The generated marketing copy should be compelling and persuasive, effectively communicating the product's value proposition to potential customers."""

validation_report = """{
  "constraint_requirement": "The generated marketing copy should be compelling and persuasive, effectively communicating the product's value proposition to potential customers.",
  "constraint_name": "marketing_copy_quality",
  "is_valid": true,
  "failure_cause": null,
  "failure_trackback": null,
  "error_type": null,
  "error_trackback": null
}"""

example: ICLExample = {
    "constraint_requirement": constraint_requirement,
    "validation_report": validation_report,
}
