from .._types import ICLExample

# Example 1: Code validation case
# This example demonstrates a requirement that can be validated deterministically.

constraint_requirement = """Don't mention the word "water"."""

validation_report = """{
  "constraint_requirement": "Don't mention the word \"water\".",
  "constraint_name": "no_water",
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
