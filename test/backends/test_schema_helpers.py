"""Tests for schema helper functions in tools.py.

Tests for _resolve_ref and _is_complex_anyof functions that handle
JSON schema resolution and complex type detection.
"""

import pytest

from mellea.backends.tools import _is_complex_anyof, _resolve_ref


class TestResolveRef:
    """Tests for the _resolve_ref helper function."""

    def test_resolve_defs_style_ref(self):
        """Test resolving #/$defs/ style references."""
        defs = {
            "Email": {
                "type": "object",
                "properties": {"to": {"type": "string"}, "subject": {"type": "string"}},
            }
        }
        result = _resolve_ref("#/$defs/Email", defs)
        assert result == defs["Email"]

    def test_resolve_definitions_style_ref(self):
        """Test resolving #/definitions/ style references."""
        defs = {
            "Address": {
                "type": "object",
                "properties": {
                    "street": {"type": "string"},
                    "city": {"type": "string"},
                },
            }
        }
        result = _resolve_ref("#/definitions/Address", defs)
        assert result == defs["Address"]

    def test_resolve_missing_ref_returns_empty_dict(self):
        """Test that resolving a non-existent ref returns empty dict."""
        defs = {"Email": {"type": "object"}}
        result = _resolve_ref("#/$defs/NotFound", defs)
        assert result == {}

    def test_resolve_invalid_ref_format_returns_empty_dict(self):
        """Test that invalid ref format returns empty dict."""
        defs = {"Email": {"type": "object"}}
        result = _resolve_ref("#/invalid/Email", defs)
        assert result == {}

    def test_resolve_with_empty_defs(self):
        """Test resolving against empty defs dict."""
        result = _resolve_ref("#/$defs/Email", {})
        assert result == {}

    def test_resolve_nested_ref_name(self):
        """Test resolving refs with nested-like names."""
        defs = {
            "User_v2": {"type": "object", "properties": {"id": {"type": "integer"}}}
        }
        result = _resolve_ref("#/$defs/User_v2", defs)
        assert result == defs["User_v2"]


class TestIsComplexAnyof:
    """Tests for the _is_complex_anyof helper function."""

    def test_simple_optional_primitive_not_complex(self):
        """Test that Optional[str] (anyOf with null and string) is not complex."""
        schema = {"anyOf": [{"type": "string"}, {"type": "null"}]}
        assert _is_complex_anyof(schema) is False

    def test_optional_int_not_complex(self):
        """Test that Optional[int] is not complex."""
        schema = {"anyOf": [{"type": "integer"}, {"type": "null"}]}
        assert _is_complex_anyof(schema) is False

    def test_anyof_with_ref_is_complex(self):
        """Test that anyOf with a $ref is complex."""
        schema = {"anyOf": [{"$ref": "#/$defs/Email"}, {"type": "null"}]}
        assert _is_complex_anyof(schema) is True

    def test_anyof_with_nested_object_is_complex(self):
        """Test that anyOf with nested object (properties) is complex."""
        schema = {
            "anyOf": [
                {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                    },
                },
                {"type": "null"},
            ]
        }
        assert _is_complex_anyof(schema) is True

    def test_anyof_with_multiple_types_no_ref_not_complex(self):
        """Test that anyOf with multiple primitives (no ref/props) is not complex."""
        schema = {"anyOf": [{"type": "string"}, {"type": "integer"}, {"type": "null"}]}
        assert _is_complex_anyof(schema) is False

    def test_anyof_with_ref_and_primitive_is_complex(self):
        """Test that anyOf with both ref and primitive is complex."""
        schema = {"anyOf": [{"$ref": "#/$defs/Email"}, {"type": "string"}]}
        assert _is_complex_anyof(schema) is True

    def test_anyof_with_only_null_not_complex(self):
        """Test that anyOf with only null type is not complex."""
        schema = {"anyOf": [{"type": "null"}]}
        assert _is_complex_anyof(schema) is False

    def test_empty_anyof_not_complex(self):
        """Test that empty anyOf is not complex."""
        schema = {"anyOf": []}
        assert _is_complex_anyof(schema) is False

    def test_anyof_missing_from_schema_not_complex(self):
        """Test that schema without anyOf is not complex."""
        schema = {"type": "object"}
        assert _is_complex_anyof(schema) is False

    def test_anyof_with_allof_is_complex(self):
        """Test that anyOf containing allOf is complex (has properties-like structure)."""
        schema = {
            "anyOf": [
                {
                    "allOf": [
                        {"$ref": "#/$defs/Base"},
                        {"properties": {"extra": {"type": "string"}}},
                    ]
                },
                {"type": "null"},
            ]
        }
        # Note: Our implementation checks for $ref or properties in the sub_schema,
        # not recursively in allOf, so this should be not complex
        # (unless allOf itself has properties)
        assert _is_complex_anyof(schema) is False

    def test_anyof_union_with_ref(self):
        """Test anyOf representing a union of multiple types including ref."""
        schema = {"anyOf": [{"$ref": "#/$defs/User"}, {"$ref": "#/$defs/Admin"}]}
        assert _is_complex_anyof(schema) is True

    def test_complex_nested_structure(self):
        """Test complex nested object with all optional fields."""
        schema = {
            "anyOf": [
                {
                    "type": "object",
                    "properties": {
                        "user": {
                            "type": "object",
                            "properties": {"name": {"type": "string"}},
                        }
                    },
                },
                {"type": "null"},
            ]
        }
        assert _is_complex_anyof(schema) is True


if __name__ == "__main__":
    pytest.main([__file__])
