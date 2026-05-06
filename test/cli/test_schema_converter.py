"""Unit tests for JSON Schema to Pydantic conversion."""

import pytest

from cli.serve.schema_converter import json_schema_to_pydantic


def test_json_schema_supports_enum_field():
    """Test that enum constraints are converted to a narrower Pydantic type."""
    model = json_schema_to_pydantic(
        {
            "type": "object",
            "properties": {"status": {"type": "string", "enum": ["open", "closed"]}},
            "required": ["status"],
        },
        "EnumExample",
    )

    parsed = model.model_validate({"status": "open"})
    assert parsed.model_dump()["status"] == "open"

    with pytest.raises(Exception):
        model.model_validate({"status": "pending"})


def test_json_schema_supports_nested_object_field():
    """Test that nested object schemas are converted recursively."""
    model = json_schema_to_pydantic(
        {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                    },
                    "required": ["name"],
                    "additionalProperties": False,
                }
            },
            "required": ["user"],
        },
        "NestedObjectExample",
    )

    parsed = model.model_validate({"user": {"name": "Alice", "age": 30}})
    parsed_user = parsed.model_dump()["user"]
    assert parsed_user["name"] == "Alice"
    assert parsed_user["age"] == 30

    with pytest.raises(Exception):
        model.model_validate({"user": {"name": "Alice", "extra": True}})


def test_json_schema_supports_array_items_schema():
    """Test that arrays validate their item schemas."""
    model = json_schema_to_pydantic(
        {
            "type": "object",
            "properties": {"tags": {"type": "array", "items": {"type": "string"}}},
            "required": ["tags"],
        },
        "ArrayExample",
    )

    parsed = model.model_validate({"tags": ["a", "b"]})
    assert parsed.model_dump()["tags"] == ["a", "b"]

    with pytest.raises(Exception):
        model.model_validate({"tags": ["a", 1]})


def test_json_schema_supports_top_level_ref():
    """Test that local refs are resolved from $defs."""
    model = json_schema_to_pydantic(
        {
            "type": "object",
            "$defs": {
                "User": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                }
            },
            "properties": {"user": {"$ref": "#/$defs/User"}},
            "required": ["user"],
        },
        "RefExample",
    )

    parsed = model.model_validate({"user": {"name": "Alice"}})
    assert parsed.model_dump()["user"]["name"] == "Alice"

    with pytest.raises(Exception):
        model.model_validate({"user": {}})


def test_json_schema_supports_anyof_field():
    """Test that representable anyOf branches are converted to unions."""
    model = json_schema_to_pydantic(
        {
            "type": "object",
            "properties": {
                "value": {"anyOf": [{"type": "string"}, {"type": "integer"}]}
            },
            "required": ["value"],
        },
        "AnyOfExample",
    )

    parsed_string = model.model_validate({"value": "hello"})
    assert parsed_string.model_dump()["value"] == "hello"

    parsed_integer = model.model_validate({"value": 7})
    assert parsed_integer.model_dump()["value"] == 7

    with pytest.raises(Exception):
        model.model_validate({"value": True})


def test_json_schema_supports_allof_object_merge():
    """Test that allOf merges object fragments into one model."""
    model = json_schema_to_pydantic(
        {
            "type": "object",
            "properties": {
                "user": {
                    "allOf": [
                        {
                            "type": "object",
                            "properties": {"name": {"type": "string"}},
                            "required": ["name"],
                        },
                        {
                            "type": "object",
                            "properties": {"age": {"type": "integer"}},
                            "required": ["age"],
                            "additionalProperties": False,
                        },
                    ]
                }
            },
            "required": ["user"],
        },
        "AllOfExample",
    )

    parsed = model.model_validate({"user": {"name": "Alice", "age": 30}})
    parsed_user = parsed.model_dump()["user"]
    assert parsed_user["name"] == "Alice"
    assert parsed_user["age"] == 30

    with pytest.raises(Exception):
        model.model_validate({"user": {"name": "Alice"}})

    with pytest.raises(Exception):
        model.model_validate({"user": {"name": "Alice", "age": 30, "extra": True}})


def test_json_schema_supports_additional_properties_schema_map():
    """Test schema-valued additionalProperties as a typed dict field."""
    model = json_schema_to_pydantic(
        {
            "type": "object",
            "properties": {
                "metadata": {
                    "type": "object",
                    "additionalProperties": {"type": "integer"},
                }
            },
            "required": ["metadata"],
        },
        "AdditionalPropertiesMapExample",
    )

    parsed = model.model_validate({"metadata": {"a": 1, "b": 2}})
    assert parsed.model_dump()["metadata"] == {"a": 1, "b": 2}

    with pytest.raises(Exception):
        model.model_validate({"metadata": {"a": "bad"}})


def test_json_schema_supports_nested_ref_in_array_items():
    """Test local refs nested under array items."""
    model = json_schema_to_pydantic(
        {
            "type": "object",
            "$defs": {
                "Tag": {
                    "type": "object",
                    "properties": {"label": {"type": "string"}},
                    "required": ["label"],
                    "additionalProperties": False,
                }
            },
            "properties": {"tags": {"type": "array", "items": {"$ref": "#/$defs/Tag"}}},
            "required": ["tags"],
        },
        "NestedRefArrayExample",
    )

    parsed = model.model_validate({"tags": [{"label": "alpha"}]})
    assert parsed.model_dump()["tags"][0]["label"] == "alpha"

    with pytest.raises(Exception):
        model.model_validate({"tags": [{"label": "alpha", "extra": True}]})


def test_json_schema_rejects_missing_type_on_property():
    """Test that properties without explicit type raise ValueError."""
    with pytest.raises(
        ValueError,
        match=r"schema must have a 'type' keyword.*not supported by this converter",
    ):
        json_schema_to_pydantic(
            {
                "type": "object",
                "properties": {"data": {"description": "anything"}},
                "required": ["data"],
            }
        )


def test_json_schema_rejects_missing_type_on_nested_object():
    """Test that nested objects without type raise ValueError."""
    with pytest.raises(
        ValueError,
        match=r"schema must have a 'type' keyword.*not supported by this converter",
    ):
        json_schema_to_pydantic(
            {
                "type": "object",
                "properties": {
                    "user": {
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"],
                    }
                },
                "required": ["user"],
            }
        )


def test_json_schema_rejects_missing_type_on_array_items():
    """Test that array items without type raise ValueError."""
    with pytest.raises(
        ValueError,
        match=r"schema must have a 'type' keyword.*not supported by this converter",
    ):
        json_schema_to_pydantic(
            {
                "type": "object",
                "properties": {
                    "items": {"type": "array", "items": {"description": "any item"}}
                },
                "required": ["items"],
            }
        )


def test_json_schema_rejects_missing_type_in_anyof_branch():
    """Test that anyOf branches without type raise ValueError."""
    with pytest.raises(
        ValueError,
        match=r"schema must have a 'type' keyword.*not supported by this converter",
    ):
        json_schema_to_pydantic(
            {
                "type": "object",
                "properties": {
                    "value": {
                        "anyOf": [{"type": "string"}, {"description": "anything"}]
                    }
                },
                "required": ["value"],
            }
        )


def test_json_schema_allows_missing_type_in_allof_branches():
    """Test that allOf branches without type default to object (intentional)."""
    # allOf is specifically for merging object fragments, so missing type
    # defaults to "object" rather than raising an error
    model = json_schema_to_pydantic(
        {
            "type": "object",
            "properties": {
                "user": {
                    "allOf": [
                        {
                            "properties": {"name": {"type": "string"}},
                            "required": ["name"],
                        },
                        {
                            "properties": {"age": {"type": "integer"}},
                            "required": ["age"],
                        },
                    ]
                }
            },
            "required": ["user"],
        }
    )

    parsed = model.model_validate({"user": {"name": "Alice", "age": 30}})
    parsed_user = parsed.model_dump()["user"]
    assert parsed_user["name"] == "Alice"
    assert parsed_user["age"] == 30
