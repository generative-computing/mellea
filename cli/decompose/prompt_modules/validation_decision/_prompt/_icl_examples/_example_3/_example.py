from .._types import ICLExample

# Example 3: Code validation case
# This example demonstrates a requirement that involves structured data validation

requirement = """The API response must conform to the following JSON schema:
{
  "type": "object",
  "properties": {
    "id": {"type": "integer"},
    "name": {"type": "string"},
    "email": {"type": "string", "format": "email"}
  },
  "required": ["id", "name", "email"]
}"""

reasoning = """This requirement specifies a precise JSON schema that the API response must follow. It can be validated deterministically by checking if the response matches the defined schema structure, data types, and required fields. This is a clear case for code validation as it involves structured data validation."""

decision = "code"

example: ICLExample = {
    "requirement": requirement,
    "reasoning": reasoning,
    "decision": decision,
}
