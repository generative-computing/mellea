from .._types import ICLExample

# Example 3: Code validation case
# This example demonstrates a requirement that involves structured data validation.

constraint_requirement = """The API response must conform to the following JSON schema:
{
  "type": "object",
  "properties": {
    "id": {"type": "integer"},
    "name": {"type": "string"},
    "email": {"type": "string", "format": "email"}
  },
  "required": ["id", "name", "email"]
}"""

validation_report = """{
  "constraint_requirement": "The API response must conform to the following JSON schema: { \\\"type\\\": \\\"object\\\", \\\"properties\\\": { \\\"id\\\": {\\\"type\\\": \\\"integer\\\"}, \\\"name\\\": {\\\"type\\\": \\\"string\\\"}, \\\"email\\\": {\\\"type\\\": \\\"string\\\", \\\"format\\\": \\\"email\\\"} }, \\\"required\\\": [\\\"id\\\", \\\"name\\\", \\\"email\\\"] }",
  "constraint_name": "api_response_schema",
  "is_valid": false,
  "failure_cause": "Response is missing required property 'email'.",
  "failure_trackback": ["Required property 'email' not found in response."],
  "error_type": null,
  "error_trackback": null
}"""

example: ICLExample = {
    "constraint_requirement": constraint_requirement,
    "validation_report": validation_report,
}
