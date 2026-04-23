"""Tests for tools with Pydantic BaseModel parameters.

This test file addresses the issue where tools defined with Pydantic BaseModel
parameters don't properly validate/coerce arguments from LLM responses.
"""

import pytest
from pydantic import BaseModel

from mellea.backends.tools import MelleaTool, validate_tool_arguments
from mellea.core import ModelToolCall

# ============================================================================
# Test Fixtures - Pydantic Models
# ============================================================================


class Email(BaseModel):
    """An email message."""

    to: str
    subject: str
    body: str


class Address(BaseModel):
    """A physical address."""

    street: str
    city: str
    state: str
    zip_code: str


class Person(BaseModel):
    """A person with contact info."""

    name: str
    age: int
    email: str
    address: Address | None = None


# ============================================================================
# Test Fixtures - Tool Functions (for schema generation testing)
# These use Pydantic BaseModel types to test schema generation
# ============================================================================


def send_email_typed(email: Email) -> str:
    """Send an email message.

    Args:
        email: The email to send
    """
    # In practice, this receives a dict from LLM tool calls
    # but we type it as Email for schema generation
    if isinstance(email, dict):
        return f"Sent email to {email['to']} with subject '{email['subject']}'"
    return f"Sent email to {email.to} with subject '{email.subject}'"


def create_contact_typed(person: Person) -> str:
    """Create a new contact.

    Args:
        person: The person's information
    """
    if isinstance(person, dict):
        return f"Created contact for {person['name']}"
    return f"Created contact for {person.name}"


def simple_nested_typed(data: Email, priority: int = 1) -> str:
    """Tool with both BaseModel and primitive parameters.

    Args:
        data: Email data
        priority: Priority level
    """
    if isinstance(data, dict):
        return f"Priority {priority}: {data['subject']}"
    return f"Priority {priority}: {data.subject}"


# ============================================================================
# Test Fixtures - Tool Functions (for actual tool calls)
# These accept dicts as LLM tool calls provide
# ============================================================================


def send_email(email: dict) -> str:
    """Send an email message.

    Args:
        email: The email to send (dict with to, subject, body)
    """
    return f"Sent email to {email['to']} with subject '{email['subject']}'"


def create_contact(person: dict) -> str:
    """Create a new contact.

    Args:
        person: The person's information (dict with name, age, email, optional address)
    """
    return f"Created contact for {person['name']}"


def simple_nested(data: dict, priority: int = 1) -> str:
    """Tool with both BaseModel and primitive parameters.

    Args:
        data: Email data (dict with to, subject, body)
        priority: Priority level
    """
    return f"Priority {priority}: {data['subject']}"


# ============================================================================
# Test Cases
# ============================================================================


class TestPydanticParameterSchemaGeneration:
    """Test that Pydantic BaseModel parameters generate correct schemas."""

    def test_simple_basemodel_schema(self):
        """Test schema generation for simple BaseModel parameter."""
        tool = MelleaTool.from_callable(send_email_typed)
        schema = tool.as_json_tool

        # Check basic structure
        assert "function" in schema
        assert schema["function"]["name"] == "send_email_typed"

        # Check parameters
        params = schema["function"]["parameters"]
        assert "properties" in params
        assert "email" in params["properties"]

        # The email parameter should have nested properties
        email_schema = params["properties"]["email"]
        assert "type" in email_schema

        # For nested objects, we expect either:
        # 1. type: "object" with nested properties
        # 2. A reference to the Email schema
        if email_schema["type"] == "object":
            assert "properties" in email_schema
            assert "to" in email_schema["properties"]
            assert "subject" in email_schema["properties"]
            assert "body" in email_schema["properties"]

    def test_nested_basemodel_schema(self):
        """Test schema generation for nested BaseModel (Person with Address)."""
        tool = MelleaTool.from_callable(create_contact_typed)
        schema = tool.as_json_tool

        params = schema["function"]["parameters"]
        person_schema = params["properties"]["person"]

        # Person should be an object type
        if person_schema["type"] == "object":
            assert "properties" in person_schema
            assert "name" in person_schema["properties"]
            assert "age" in person_schema["properties"]
            assert "email" in person_schema["properties"]
            assert "address" in person_schema["properties"]

            # Address should also be properly nested
            address_schema = person_schema["properties"]["address"]
            if address_schema.get("type") == "object":
                assert "properties" in address_schema
                assert "street" in address_schema["properties"]

    def test_mixed_parameters_schema(self):
        """Test schema with both BaseModel and primitive parameters."""
        tool = MelleaTool.from_callable(simple_nested_typed)
        schema = tool.as_json_tool

        params = schema["function"]["parameters"]
        assert "data" in params["properties"]
        assert "priority" in params["properties"]

        # Priority should be a simple integer
        priority_schema = params["properties"]["priority"]
        assert "integer" in priority_schema.get("type", "")


class TestPydanticParameterValidation:
    """Test validation of Pydantic BaseModel parameters."""

    def test_valid_nested_object(self):
        """Test validation with correctly structured nested object."""
        tool = MelleaTool.from_callable(send_email)

        # LLM returns a properly structured email object
        args = {
            "email": {
                "to": "user@example.com",
                "subject": "Test Subject",
                "body": "Test body content",
            }
        }

        validated = validate_tool_arguments(tool, args, coerce_types=True)

        # Should validate successfully
        assert "email" in validated
        assert validated["email"]["to"] == "user@example.com"
        assert validated["email"]["subject"] == "Test Subject"
        assert validated["email"]["body"] == "Test body content"

    def test_nested_object_with_type_coercion(self):
        """Test that nested object fields can be coerced."""
        tool = MelleaTool.from_callable(create_contact_typed)

        # LLM returns age as string
        args = {
            "person": {
                "name": "John Doe",
                "age": "30",  # String instead of int
                "email": "john@example.com",
            }
        }

        validated = validate_tool_arguments(tool, args, coerce_types=True)

        # Age should be coerced to int
        assert validated["person"]["age"] == 30
        assert isinstance(validated["person"]["age"], int)

    def test_missing_required_nested_field(self):
        """Test validation fails when required nested field is missing."""
        tool = MelleaTool.from_callable(send_email_typed)

        # Missing 'body' field
        args = {"email": {"to": "user@example.com", "subject": "Test"}}

        # In lenient mode, should return original args
        validated = validate_tool_arguments(tool, args, strict=False)
        assert validated == args

        # In strict mode, should raise
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            validate_tool_arguments(tool, args, strict=True)

    def test_optional_nested_object(self):
        """Test validation with optional nested object."""
        tool = MelleaTool.from_callable(create_contact_typed)

        # Address is optional, so this should be valid
        args = {"person": {"name": "Jane Doe", "age": 25, "email": "jane@example.com"}}

        validated = validate_tool_arguments(tool, args, coerce_types=True)
        assert validated["person"]["name"] == "Jane Doe"
        assert (
            "address" not in validated["person"]
            or validated["person"].get("address") is None
        )

    def test_nested_object_with_all_fields(self):
        """Test validation with all fields including optional nested object."""
        tool = MelleaTool.from_callable(create_contact_typed)

        args = {
            "person": {
                "name": "Bob Smith",
                "age": 35,
                "email": "bob@example.com",
                "address": {
                    "street": "123 Main St",
                    "city": "Boston",
                    "state": "MA",
                    "zip_code": "02101",
                },
            }
        }

        validated = validate_tool_arguments(tool, args, coerce_types=True)
        assert validated["person"]["address"]["city"] == "Boston"


class TestPydanticParameterToolCall:
    """Test actual tool calls with Pydantic BaseModel parameters."""

    def test_tool_call_with_basemodel(self):
        """Test that tool can be called with validated BaseModel args."""
        tool = MelleaTool.from_callable(send_email)

        args = {
            "email": {
                "to": "test@example.com",
                "subject": "Hello",
                "body": "Test message",
            }
        }

        validated = validate_tool_arguments(tool, args, coerce_types=True)
        tool_call = ModelToolCall("send_email", tool, validated)
        result = tool_call.call_func()

        assert "test@example.com" in result
        assert "Hello" in result

    def test_tool_call_with_coerced_nested_types(self):
        """Test tool call with type coercion in nested object."""
        tool = MelleaTool.from_callable(create_contact)

        # Age as string, should be coerced
        args = {"person": {"name": "Alice", "age": "28", "email": "alice@example.com"}}

        validated = validate_tool_arguments(tool, args, coerce_types=True)
        tool_call = ModelToolCall("create_contact", tool, validated)
        result = tool_call.call_func()

        assert "Alice" in result

    def test_mixed_parameters_tool_call(self):
        """Test tool with both BaseModel and primitive parameters."""
        tool = MelleaTool.from_callable(simple_nested_typed)

        args = {
            "data": {
                "to": "user@example.com",
                "subject": "Important",
                "body": "Message",
            },
            "priority": "5",  # String that should be coerced to int
        }

        validated = validate_tool_arguments(tool, args, coerce_types=True)
        assert validated["priority"] == 5
        assert isinstance(validated["priority"], int)

        tool_call = ModelToolCall("simple_nested", tool, validated)
        result = tool_call.call_func()

        assert "Priority 5" in result
        assert "Important" in result


class TestEdgeCases:
    """Test edge cases with Pydantic parameters."""

    def test_flat_dict_instead_of_nested(self):
        """Test when LLM returns flat dict instead of nested structure."""
        tool = MelleaTool.from_callable(send_email_typed)

        # LLM might incorrectly flatten the structure
        args = {"to": "user@example.com", "subject": "Test", "body": "Content"}

        # This should fail validation in strict mode
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            validate_tool_arguments(tool, args, strict=True)

        # In lenient mode, returns original
        validated = validate_tool_arguments(tool, args, strict=False)
        assert validated == args

    def test_extra_fields_in_nested_object(self):
        """Test nested object with extra fields not in schema."""
        tool = MelleaTool.from_callable(send_email_typed)

        args = {
            "email": {
                "to": "user@example.com",
                "subject": "Test",
                "body": "Content",
                "extra_field": "should be ignored or preserved",
            }
        }

        # In lenient mode, extra fields might be preserved
        validated = validate_tool_arguments(tool, args, strict=False)
        assert validated["email"]["to"] == "user@example.com"


class TestOptionalParameterRegression:
    """Test cases to prevent regression of Optional parameter handling.

    These tests verify the fix for the anyOf narrowing issue where Optional
    parameters were incorrectly treated as complex types and added to required.
    """

    def test_basemodel_param_inlined_no_ref(self):
        """Test def f(email: Email) — required BaseModel param.

        Confirms the core fix works: email is inlined in the schema (no $ref in
        the output), and validate_tool_arguments accepts the dict without error.
        """

        def send_email(email: Email) -> str:
            """Send an email.

            Args:
                email: The email to send
            """
            return f"Sent to {email.to}"

        tool = MelleaTool.from_callable(send_email)
        schema = tool.as_json_tool

        # Verify email is inlined (no $ref)
        params = schema["function"]["parameters"]
        email_prop = params["properties"]["email"]
        assert "$ref" not in email_prop, "Email should be inlined, not a $ref"
        assert email_prop["type"] == "object"
        assert "properties" in email_prop
        assert "to" in email_prop["properties"]
        assert "subject" in email_prop["properties"]
        assert "body" in email_prop["properties"]

        # Verify email is required
        assert "email" in params["required"]

        # Verify validation works
        args = {"email": {"to": "a@b.com", "subject": "hi", "body": "test"}}
        validated = validate_tool_arguments(tool, args, coerce_types=True)
        assert validated["email"]["to"] == "a@b.com"
        assert validated["email"]["subject"] == "hi"

    def test_optional_scalar_not_required(self):
        """Test def f(x: str, y: str | None = None) — Optional scalar.

        Confirms Optional params still work: y must be absent from required and
        the schema type must be "string", not a raw anyOf structure.
        """

        def process_text(x: str, y: str | None = None) -> str:
            """Process text with optional parameter.

            Args:
                x: Required text
                y: Optional additional text
            """
            return f"{x} {y or ''}"

        tool = MelleaTool.from_callable(process_text)
        schema = tool.as_json_tool

        params = schema["function"]["parameters"]

        # Verify x is required
        assert "x" in params["required"]

        # Verify y is NOT required
        assert "y" not in params["required"], (
            "Optional parameter y should not be in required"
        )

        # Verify y has simple string type, not raw anyOf
        y_prop = params["properties"]["y"]
        assert y_prop["type"] == "string", (
            f"Expected 'string', got {y_prop.get('type')}"
        )
        assert "anyOf" not in y_prop, (
            "Optional scalar should be flattened, not preserve anyOf"
        )

        # Verify validation works with y absent
        args = {"x": "hello"}
        validated = validate_tool_arguments(tool, args, coerce_types=True)
        assert validated["x"] == "hello"

        # Verify validation works with y present
        args = {"x": "hello", "y": "world"}
        validated = validate_tool_arguments(tool, args, coerce_types=True)
        assert validated["x"] == "hello"
        assert validated["y"] == "world"

    @pytest.mark.skip(
        reason="Nested model resolution not yet implemented. "
        "This test documents the expected behavior once recursive $ref resolution is added. "
        "Currently fails because Address remains as a dangling $ref inside Person's schema. "
        "NESTED_MODEL_RESOLUTION_ISSUE.md in "
        "https://github.com/generative-computing/mellea/issues/911 for implementation details."
    )
    def test_nested_models_fully_inlined(self):
        """Test def f(person: Person) where Person has address: Address.

        Confirms nested models work end-to-end: both Person and Address fully
        inlined in the schema, and validate_tool_arguments accepts a nested
        dict without a ValidationError.

        DISABLED: This test is currently skipped because nested model resolution
        is not yet implemented. The test will be enabled once the recursive
        reference resolution feature is added to convert_function_to_ollama_tool.
        """

        def create_person(person: Person) -> str:
            """Create a person record.

            Args:
                person: The person's information
            """
            return f"Created {person.name}"

        tool = MelleaTool.from_callable(create_person)
        schema = tool.as_json_tool

        params = schema["function"]["parameters"]
        person_prop = params["properties"]["person"]

        # Verify Person is inlined
        assert "$ref" not in person_prop, "Person should be inlined, not a $ref"
        assert person_prop["type"] == "object"
        assert "properties" in person_prop

        # Verify Person has all expected fields
        person_props = person_prop["properties"]
        assert "name" in person_props
        assert "age" in person_props
        assert "email" in person_props
        assert "address" in person_props

        # Verify Address is also inlined (not a dangling $ref)
        address_prop = person_props["address"]
        # Address is Optional[Address], so it might be in anyOf
        if "anyOf" in address_prop:
            # Find the non-null schema in anyOf
            address_schemas = [
                s for s in address_prop["anyOf"] if s.get("type") != "null"
            ]
            assert len(address_schemas) > 0, "Should have at least one non-null schema"
            address_schema = address_schemas[0]
        else:
            address_schema = address_prop

        # The address schema should be fully resolved (no $ref)
        assert "$ref" not in address_schema, (
            "Address should be inlined, not a dangling $ref"
        )

        # Note: Current implementation may not fully inline nested models yet
        # This test documents the expected behavior for the nested resolution issue

        # Verify validation works with nested dict
        args = {
            "person": {
                "name": "John Doe",
                "age": 30,
                "email": "john@example.com",
                "address": {
                    "street": "123 Main St",
                    "city": "Boston",
                    "state": "MA",
                    "zip_code": "02101",
                },
            }
        }

        # This should not raise ValidationError
        validated = validate_tool_arguments(tool, args, coerce_types=True)
        assert validated["person"]["name"] == "John Doe"
        assert validated["person"]["address"]["city"] == "Boston"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
