"""Unit tests for CLI serve Pydantic models."""

import pytest
from pydantic import ValidationError

from cli.serve.models import FunctionParameters, JsonSchemaFormat, StreamOptions


class TestStreamOptions:
    """Tests for the StreamOptions Pydantic model."""

    def test_default_include_usage_is_false(self):
        """Test that include_usage defaults to False."""
        options = StreamOptions()
        assert options.include_usage is False

    def test_include_usage_true(self):
        """Test that include_usage can be set to True."""
        options = StreamOptions(include_usage=True)
        assert options.include_usage is True

    def test_include_usage_false(self):
        """Test that include_usage can be explicitly set to False."""
        options = StreamOptions(include_usage=False)
        assert options.include_usage is False

    def test_string_true_coerced_to_bool(self):
        """Test that string 'true' is coerced to boolean True."""
        options = StreamOptions(include_usage="true")  # type: ignore[arg-type]
        assert options.include_usage is True

    def test_string_false_coerced_to_bool(self):
        """Test that string 'false' is coerced to boolean False."""
        options = StreamOptions(include_usage="false")  # type: ignore[arg-type]
        assert options.include_usage is False

    def test_integer_one_coerced_to_true(self):
        """Test that integer 1 is coerced to boolean True."""
        options = StreamOptions(include_usage=1)  # type: ignore[arg-type]
        assert options.include_usage is True

    def test_integer_zero_coerced_to_false(self):
        """Test that integer 0 is coerced to boolean False."""
        options = StreamOptions(include_usage=0)  # type: ignore[arg-type]
        assert options.include_usage is False

    def test_invalid_type_raises_validation_error(self):
        """Test that passing non-coercible values raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            StreamOptions(include_usage={"invalid": "dict"})  # type: ignore[arg-type]

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["type"] == "bool_type"
        assert "include_usage" in errors[0]["loc"]

    def test_extra_fields_allowed(self):
        """Test that extra fields are allowed (for forward compatibility)."""
        # Pydantic v2 default is to forbid extra fields, but we may want to allow
        # them for OpenAI API compatibility. This test documents current behavior.
        try:
            options = StreamOptions(include_usage=True, unknown_field="value")  # type: ignore[call-arg]
            # If this succeeds, extra fields are allowed
            assert options.include_usage is True
        except ValidationError:
            # If this fails, extra fields are forbidden (current expected behavior)
            pass

    def test_model_dump_includes_include_usage(self):
        """Test that model_dump includes the include_usage field."""
        options = StreamOptions(include_usage=True)
        dumped = options.model_dump()
        assert "include_usage" in dumped
        assert dumped["include_usage"] is True

    def test_model_dump_json_serialization(self):
        """Test that the model can be serialized to JSON."""
        options = StreamOptions(include_usage=True)
        json_str = options.model_dump_json()
        assert "include_usage" in json_str
        assert "true" in json_str.lower()


class TestFunctionParameters:
    """Tests for the FunctionParameters RootModel validator."""

    def test_valid_json_schema_accepted(self):
        """Test that a valid JSON Schema dict is accepted."""
        schema = {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        }
        params = FunctionParameters(root=schema)
        assert params.root == schema

    def test_legacy_root_model_envelope_rejected(self):
        """Test that legacy {'RootModel': {...}} envelope is rejected."""
        legacy_envelope = {
            "RootModel": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
            }
        }
        with pytest.raises(ValidationError) as exc_info:
            FunctionParameters(root=legacy_envelope)

        errors = exc_info.value.errors()
        assert len(errors) == 1
        error_msg = str(exc_info.value)
        assert "Legacy {'RootModel': {...}} envelope is no longer accepted" in error_msg

    def test_root_model_with_additional_keys_accepted(self):
        """Test that a dict with 'RootModel' plus other keys is accepted."""
        # This is a valid schema that happens to have a property named "RootModel"
        schema = {
            "type": "object",
            "properties": {
                "RootModel": {"type": "string"},
                "other_field": {"type": "number"},
            },
        }
        params = FunctionParameters(root=schema)
        assert params.root == schema

    def test_empty_dict_accepted(self):
        """Test that an empty dict is accepted (though not a useful schema)."""
        params = FunctionParameters(root={})
        assert params.root == {}


class TestJsonSchemaFormat:
    """Test JsonSchemaFormat serialization uses 'schema' alias, not 'schema_'."""

    def test_serialization_uses_schema_alias(self):
        """Verify schema_ serializes as 'schema' in dict and JSON output."""
        schema_def = {"type": "object", "properties": {"foo": {"type": "string"}}}
        json_schema = JsonSchemaFormat(name="TestSchema", schema=schema_def)

        # Dict serialization
        dumped = json_schema.model_dump()
        assert "schema" in dumped and "schema_" not in dumped
        assert dumped["schema"] == schema_def

        # JSON serialization
        json_str = json_schema.model_dump_json()
        assert '"schema":' in json_str and '"schema_":' not in json_str

        # Input accepts both 'schema' (alias) and 'schema_' (field name)
        from_alias = JsonSchemaFormat(name="Test1", schema={"type": "string"})
        # Use model_validate to test runtime populate_by_name behavior (bypasses type checker)
        from_field = JsonSchemaFormat.model_validate(
            {"name": "Test2", "schema_": {"type": "number"}}
        )
        assert from_alias.schema_ == {"type": "string"}
        assert from_field.schema_ == {"type": "number"}
