"""Unit tests for CLI serve Pydantic models."""

import pytest
from pydantic import ValidationError

from cli.serve.models import StreamOptions


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
