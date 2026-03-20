from .._types import ICLExample

# Example 2: LLM validation case
# This example demonstrates a requirement that needs subjective evaluation.

constraint_requirement = """The user interface should be intuitive and provide a seamless experience for first-time users."""

validation_report = """{
  "constraint_requirement": "The user interface should be intuitive and provide a seamless experience for first-time users.",
  "constraint_name": "intuitive_user_experience",
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
