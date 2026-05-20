# pytest: skip_always
"""Client demonstrating response_format parameter with m serve.

This example shows how to use the three response_format types:
1. text - Plain text output (default)
2. json_object - Unstructured JSON output
3. json_schema - Structured output validated against a JSON schema

Prerequisites:
    Start the server first:
        m serve docs/examples/m_serve/m_serve_example_response_format.py

    Then run this client:
        python docs/examples/m_serve/client_response_format.py
"""

import json

import openai

PORT = 8080
BASE_URL = f"http://0.0.0.0:{PORT}/v1"

# Create OpenAI client pointing to our m serve endpoint
client = openai.OpenAI(api_key="not-needed", base_url=BASE_URL)


def example_text_format():
    """Example 1: Plain text output (default behavior)."""
    print("\n" + "=" * 60)
    print("Example 1: Text Format (default)")
    print("=" * 60)

    response = client.chat.completions.create(
        model="granite4:micro-h",
        messages=[{"role": "user", "content": "Write a haiku about programming."}],
        response_format={"type": "text"},
    )

    print(f"Response: {response.choices[0].message.content}")


def example_json_object():
    """Example 2: Unstructured JSON output.

    Note: json_object format requests JSON but doesn't enforce it strictly.
    The model may wrap JSON in markdown or add explanatory text.
    For strict JSON validation, use json_schema instead.
    """
    print("\n" + "=" * 60)
    print("Example 2: JSON Object Format")
    print("=" * 60)

    response = client.chat.completions.create(
        model="granite4:micro-h",
        messages=[
            {
                "role": "user",
                "content": "Generate a JSON object with information about a fictional person. Include name, age, and city. Return ONLY the JSON, no markdown formatting.",
            }
        ],
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content or ""
    print(f"Response: {content}")

    # First, try to parse as-is (valid JSON)
    try:
        data = json.loads(content)
        print("\n✓ Valid JSON received")
        print(f"\nParsed JSON:\n{json.dumps(data, indent=2)}")
        return
    except json.JSONDecodeError:
        # Not valid JSON, try to extract from markdown
        print("\n⚠ Response is not valid JSON, attempting to extract from markdown...")

    # Fallback: Try to extract JSON from markdown code blocks
    json_content = content
    if "```json" in content:
        # Extract JSON from markdown code block
        start = content.find("```json") + 7
        end = content.find("```", start)
        if end > start:
            json_content = content[start:end].strip()
            print("Extracted from ```json block")
    elif "```" in content:
        # Generic code block
        start = content.find("```") + 3
        end = content.find("```", start)
        if end > start:
            json_content = content[start:end].strip()
            print("Extracted from ``` block")

    # Try parsing the extracted content
    try:
        data = json.loads(json_content)
        print(
            f"\n✓ Successfully extracted and parsed JSON:\n{json.dumps(data, indent=2)}"
        )
    except json.JSONDecodeError as e:
        print("\n✗ Failed to parse JSON even after extraction")
        print("Note: json_object format doesn't enforce strict JSON.")
        print("For guaranteed JSON output, use json_schema format instead.")
        print(f"Parse error: {e}")


def example_json_schema_person():
    """Example 3: Structured output with JSON schema validation."""
    print("\n" + "=" * 60)
    print("Example 3: JSON Schema Format - Person")
    print("=" * 60)

    # Define a JSON schema for a person
    person_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "The person's full name"},
            "age": {"type": "integer", "description": "The person's age in years"},
            "email": {"type": "string", "description": "The person's email address"},
            "city": {
                "type": "string",
                "description": "The city where the person lives",
            },
        },
        "required": ["name", "age", "email"],
        "additionalProperties": False,
    }

    response = client.chat.completions.create(
        model="granite4:micro-h",
        messages=[
            {
                "role": "user",
                "content": "Generate information about a software engineer named Alice.",
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "Person", "schema": person_schema, "strict": True},
        },
    )

    content = response.choices[0].message.content
    print(f"Response: {content}")

    # Parse and validate the structured output
    try:
        data = json.loads(content or "{}")
        print(f"\nParsed structured output:\n{json.dumps(data, indent=2)}")

        # Verify required fields
        assert "name" in data, "Missing required field: name"
        assert "age" in data, "Missing required field: age"
        assert "email" in data, "Missing required field: email"
        print("\n✓ All required fields present")

    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON: {e}")
    except AssertionError as e:
        print(f"Validation error: {e}")


def example_json_schema_product():
    """Example 4: Structured output for a product catalog."""
    print("\n" + "=" * 60)
    print("Example 4: JSON Schema Format - Product")
    print("=" * 60)

    # Define a JSON schema for a product
    product_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Product name"},
            "price": {"type": "number", "description": "Price in USD"},
            "category": {
                "type": "string",
                "enum": ["electronics", "clothing", "food", "books"],
                "description": "Product category",
            },
            "in_stock": {
                "type": "boolean",
                "description": "Whether the product is in stock",
            },
            "description": {"type": "string", "description": "Product description"},
        },
        "required": ["name", "price", "category", "in_stock"],
        "additionalProperties": False,
    }

    response = client.chat.completions.create(
        model="granite4:micro-h",
        messages=[
            {
                "role": "user",
                "content": "Generate a product listing for a laptop computer.",
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "Product",
                "schema": product_schema,
                "strict": True,
            },
        },
    )

    content = response.choices[0].message.content
    print(f"Response: {content}")

    # Parse and display the structured output
    try:
        data = json.loads(content or "{}")
        print(f"\nParsed product data:\n{json.dumps(data, indent=2)}")

        # Verify the category is valid
        valid_categories = ["electronics", "clothing", "food", "books"]
        if data.get("category") in valid_categories:
            print(f"\n✓ Valid category: {data['category']}")

    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON: {e}")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("RESPONSE_FORMAT EXAMPLES")
    print("=" * 60)
    print(f"Connecting to: {BASE_URL}")
    print("=" * 60)

    try:
        # Run all examples
        example_text_format()
        example_json_object()
        example_json_schema_person()
        example_json_schema_product()

        print("\n" + "=" * 60)
        print("ALL EXAMPLES COMPLETED")
        print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure the server is running:")
        print(
            f"  m serve docs/examples/m_serve/m_serve_example_response_format.py --port {PORT}"
        )


if __name__ == "__main__":
    main()
