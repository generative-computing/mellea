from .._types import ICLExample

# Example 5: LLM validation case
# This example demonstrates a requirement that involves semantic content interpretation
# and structural elements in test case descriptions.

constraint_requirement = """Each test case should include: Test Case description, Precondition, Test Steps, and Expected Outcome"""

validation_report = """{
  "constraint_requirement": "Each test case should include: Test Case description, Precondition, Test Steps, and Expected Outcome",
  "constraint_name": "test_case_structure",
  "is_valid": false,
  "failure_cause": "The 'Expected Outcome' section is missing from the test case.",
  "failure_trackback": ["Expected Outcome section not found."],
  "error_type": null,
  "error_trackback": null
}"""

example: ICLExample = {
    "constraint_requirement": constraint_requirement,
    "validation_report": validation_report,
}
