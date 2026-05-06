"""Helpers for converting OpenAI-style JSON Schema response formats."""

from enum import Enum
from typing import Annotated, Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Strict, create_model


def json_schema_to_pydantic(
    schema: dict[str, Any], model_name: str = "DynamicModel"
) -> type[BaseModel]:
    """Convert a practical subset of JSON Schema to a Pydantic model dynamically.

    This converter targets the OpenAI-style structured output schemas used by
    ``m serve``. It intentionally maps JSON Schema features into Python typing
    and Pydantic model semantics rather than attempting to preserve every JSON
    Schema validation rule exactly.

    Supported features:
    - top-level and nested ``object`` schemas with ``properties`` and ``required``
    - primitive types: ``string``, ``integer``, ``number``, ``boolean``
    - arrays via ``type: "array"`` with supported ``items``
    - string or primitive enums via ``enum``
    - nullable fields via ``type: ["<type>", "null"]``
    - local ``$ref`` into ``$defs`` / ``definitions``
    - simple ``allOf`` merging for object-like schemas
    - simple ``anyOf`` / ``oneOf`` unions when each branch is representable
    - boolean and schema-valued ``additionalProperties``

    Behavior notes:
    - ``additionalProperties: false`` maps to ``extra="forbid"``
    - ``additionalProperties: true`` maps to ``extra="ignore"``
    - schema-valued ``additionalProperties`` maps to ``dict[str, ValueType]``
      only for open-ended object maps. It cannot be combined with named
      ``properties`` because that is not representable as a single standard
      Pydantic field shape without custom validators.
    - sibling keywords next to ``$ref`` are merged over the resolved target,
      matching common JSON Schema practice for OpenAI-compatible schemas

    Still unsupported and will raise ``ValueError``:
    - non-local refs
    - recursive ``$ref`` cycles
    - tuple-style array schemas
    - object schemas without ``properties`` unless they are pure
      ``additionalProperties`` maps
    - schema constraints beyond representable typing/extra handling

    Args:
        schema: JSON Schema definition (must have top-level ``type: "object"``).
        model_name: Name for the generated Pydantic model.

    Returns:
        A dynamically created Pydantic model class.

    Raises:
        ValueError: If the schema is invalid or unsupported.
    """
    defs = schema.get("$defs")
    if defs is None:
        defs = schema.get("definitions", {})
    if defs is None:
        defs = {}
    if not isinstance(defs, dict):
        raise ValueError("Schema '$defs' must be an object")

    ref_cache: dict[str, Any] = {}
    model_cache: dict[str, type[BaseModel]] = {}
    in_flight_refs: set[str] = set()
    ref_name_by_schema_id: dict[int, str] = {}

    def _sanitize_model_name(name: str) -> str:
        sanitized = "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")
        return sanitized or "DynamicModel"

    def _format_path(path: str) -> str:
        return path or "<root>"

    def _resolve_ref(ref: str) -> tuple[str, dict[str, Any]]:
        if ref in ref_cache:
            resolved = ref_cache[ref]
            if not isinstance(resolved, dict):
                raise ValueError(f"Resolved ref is invalid: {ref}")

            for prefix in ("#/$defs/", "#/definitions/"):
                if ref.startswith(prefix):
                    return ref[len(prefix) :], resolved
            raise ValueError(
                f"Only local $ref values into $defs/definitions are supported: {ref}"
            )

        prefixes = ("#/$defs/", "#/definitions/")
        for prefix in prefixes:
            if ref.startswith(prefix):
                key = ref[len(prefix) :]
                if key not in defs:
                    raise ValueError(f"Unresolved local ref: {ref}")
                target = defs[key]
                if not isinstance(target, dict):
                    raise ValueError(f"Ref target must be an object: {ref}")
                ref_cache[ref] = target
                ref_name_by_schema_id[id(target)] = key
                return key, target

        raise ValueError(
            f"Only local $ref values into $defs/definitions are supported: {ref}"
        )

    def _merge_nullable(annotation: Any, is_nullable: bool) -> Any:
        """Wrap an annotation in ``None`` when the source schema is nullable."""
        if is_nullable:
            return annotation | None
        return annotation

    def _enum_annotation(enum_values: list[Any], path: str) -> Any:
        """Convert JSON Schema enum values into a Python typing annotation."""
        if not enum_values:
            raise ValueError(f"{_format_path(path)} enum must not be empty")

        value_types = {type(value) for value in enum_values}
        if len(value_types) != 1:
            raise ValueError(
                f"{_format_path(path)} enum values must all have the same primitive type"
            )

        value_type = value_types.pop()
        allowed_types = {str, int, float, bool}
        if value_type not in allowed_types:
            raise ValueError(
                f"{_format_path(path)} enum values must be string, integer, number, or boolean"
            )

        if value_type is str:
            enum_name = _sanitize_model_name(
                path.replace(".", "_").replace("[", "_").replace("]", "")
            )
            members = {
                (
                    value.upper() if value and value[0].isalpha() else f"VALUE_{index}"
                ): value
                for index, value in enumerate(enum_values)
            }
            return Enum(enum_name or "GeneratedEnum", members)

        return Literal[tuple(enum_values)]

    def _merge_object_schemas(
        schemas: list[dict[str, Any]], path: str
    ) -> dict[str, Any]:
        """Merge simple object schemas for ``allOf``.

        This supports the common OpenAI-compatible case where ``allOf`` is used
        to compose object fragments. Conflicting keywords are rejected rather
        than silently guessed.
        """
        merged: dict[str, Any] = {"type": "object", "properties": {}, "required": []}
        merged_required: set[str] = set()
        merged_additional_properties: bool | dict[str, Any] = True

        for index, branch in enumerate(schemas):
            resolved_branch = _normalize_schema(branch, f"{path}.allOf[{index}]")
            branch_type = resolved_branch.get("type", "object")
            if branch_type != "object":
                raise ValueError(
                    f"{_format_path(path)} allOf only supports object branches"
                )

            branch_properties = resolved_branch.get("properties", {})
            if not isinstance(branch_properties, dict):
                raise ValueError(
                    f"{_format_path(path)} allOf branch properties must be an object"
                )

            for property_name, property_schema in branch_properties.items():
                if property_name in merged["properties"]:
                    raise ValueError(
                        f"{_format_path(path)} allOf has conflicting property "
                        f"definitions for '{property_name}'"
                    )
                cast(dict[str, Any], merged["properties"])[property_name] = (
                    property_schema
                )

            branch_required = resolved_branch.get("required", [])
            if not isinstance(branch_required, list):
                raise ValueError(
                    f"{_format_path(path)} allOf branch 'required' must be an array"
                )
            merged_required.update(
                field_name
                for field_name in branch_required
                if isinstance(field_name, str)
            )

            branch_additional_properties = resolved_branch.get(
                "additionalProperties", True
            )
            if branch_additional_properties is False:
                merged_additional_properties = False
            elif isinstance(branch_additional_properties, dict):
                if merged_additional_properties is True:
                    merged_additional_properties = branch_additional_properties
                elif merged_additional_properties is False:
                    continue
                elif merged_additional_properties != branch_additional_properties:
                    raise ValueError(
                        f"{_format_path(path)} allOf has conflicting "
                        "additionalProperties schemas"
                    )

        merged["required"] = sorted(merged_required)
        merged["additionalProperties"] = merged_additional_properties
        return merged

    def _union_annotation(
        keyword: str, union_schemas: list[dict[str, Any]], path: str
    ) -> Any:
        """Convert ``anyOf``/``oneOf`` branches into a Python union annotation."""
        if not union_schemas:
            raise ValueError(f"{_format_path(path)} {keyword} must not be empty")

        annotations: list[Any] = []
        for index, branch in enumerate(union_schemas):
            annotations.append(
                _schema_to_type(branch, f"{path}.{keyword}[{index}]", in_union=True)
            )

        annotation = annotations[0]
        for branch_annotation in annotations[1:]:
            annotation = annotation | branch_annotation
        return annotation

    def _normalize_schema(field_schema: dict[str, Any], path: str) -> dict[str, Any]:
        """Resolve refs and simple combinators into a normalized schema object."""
        if not isinstance(field_schema, dict):
            raise ValueError(f"{_format_path(path)} schema must be an object")

        normalized = dict(field_schema)

        if "$ref" in normalized:
            ref = normalized["$ref"]
            if not isinstance(ref, str):
                raise ValueError(f"{_format_path(path)} $ref must be a string")
            _, resolved = _resolve_ref(ref)
            sibling_keywords = {k: v for k, v in normalized.items() if k != "$ref"}
            if sibling_keywords:
                merged = dict(resolved)
                merged.update(sibling_keywords)
                normalized = merged
            else:
                normalized = dict(resolved)

        if "allOf" in normalized:
            all_of = normalized.pop("allOf")
            if not isinstance(all_of, list):
                raise ValueError(f"{_format_path(path)} allOf must be an array")
            merged = _merge_object_schemas(all_of, path)
            merged.update(normalized)
            normalized = merged

        return normalized

    def _schema_to_type(
        field_schema: dict[str, Any], path: str, in_union: bool = False
    ) -> Any:
        """Convert a JSON Schema node into a Python typing annotation."""
        normalized_schema = _normalize_schema(field_schema, path)

        for keyword in ("anyOf", "oneOf"):
            if keyword in normalized_schema:
                union_schemas = normalized_schema[keyword]
                if not isinstance(union_schemas, list):
                    raise ValueError(f"{_format_path(path)} {keyword} must be an array")
                sibling_keywords = {
                    key: value
                    for key, value in normalized_schema.items()
                    if key != keyword
                }
                branch_schemas: list[dict[str, Any]] = []
                for branch in union_schemas:
                    if not isinstance(branch, dict):
                        raise ValueError(
                            f"{_format_path(path)} {keyword} branches must be objects"
                        )
                    merged_branch = dict(branch)
                    for sibling_key, sibling_value in sibling_keywords.items():
                        merged_branch.setdefault(sibling_key, sibling_value)
                    branch_schemas.append(merged_branch)
                return _union_annotation(keyword, branch_schemas, path)

        if "enum" in normalized_schema:
            enum_values = normalized_schema["enum"]
            if not isinstance(enum_values, list):
                raise ValueError(f"{_format_path(path)} enum must be an array")
            return _enum_annotation(enum_values, path)

        field_type = normalized_schema.get("type")
        if field_type is None:
            raise ValueError(
                f"{_format_path(path)} schema must have a 'type' keyword. "
                "JSON Schema without 'type' is valid but not supported by this converter. "
                "Please add an explicit type (e.g., 'string', 'integer', 'object', 'array')."
            )
        is_nullable = False
        if isinstance(field_type, list):
            non_null_types = [item for item in field_type if item != "null"]
            null_count = len(field_type) - len(non_null_types)
            if null_count > 1 or len(non_null_types) != 1:
                raise ValueError(
                    f"{_format_path(path)} uses unsupported multi-type schema: {field_type}"
                )
            if null_count == 1:
                is_nullable = True
            field_type = non_null_types[0]

        primitive_type_mapping = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
        }

        if field_type in primitive_type_mapping:
            base_type = primitive_type_mapping[field_type]
            if in_union:
                annotated_type = Annotated[base_type, Strict()]  # type: ignore[valid-type]
                return _merge_nullable(annotated_type, is_nullable)
            return _merge_nullable(base_type, is_nullable)

        if field_type == "object":
            properties = normalized_schema.get("properties")
            additional_properties = normalized_schema.get("additionalProperties", True)

            if properties is None and isinstance(additional_properties, dict):
                value_annotation = _schema_to_type(additional_properties, f"{path}.*")
                dict_type = dict[str, value_annotation]  # type: ignore[valid-type]
                return _merge_nullable(dict_type, is_nullable)

            nested_name = _sanitize_model_name(f"{model_name}_{path.replace('.', '_')}")
            nested_model = _object_schema_to_model(normalized_schema, nested_name, path)
            return _merge_nullable(nested_model, is_nullable)

        if field_type == "array":
            items_schema = normalized_schema.get("items")
            item_annotation: Any
            if items_schema is None:
                item_annotation = Any
            elif isinstance(items_schema, list):
                raise ValueError(
                    f"{_format_path(path)} uses unsupported tuple-style array schema"
                )
            elif isinstance(items_schema, dict):
                item_annotation = _schema_to_type(items_schema, f"{path}[]")
            else:
                raise ValueError(f"{_format_path(path)} items must be an object")
            # Construct list type at runtime to avoid mypy subscript error.
            list_type = list[item_annotation]  # type: ignore[valid-type]
            return _merge_nullable(list_type, is_nullable)

        raise ValueError(
            f"{_format_path(path)} uses unsupported JSON schema type: {field_type}"
        )

    def _object_schema_to_model(
        object_schema: dict[str, Any], current_model_name: str, path: str
    ) -> type[BaseModel]:
        current_ref_name = ref_name_by_schema_id.get(id(object_schema))
        if current_ref_name is not None:
            if current_ref_name in in_flight_refs:
                raise ValueError("recursive $ref is not supported")
            in_flight_refs.add(current_ref_name)

        try:
            normalized_schema = _normalize_schema(object_schema, path)
            if normalized_schema.get("type") != "object":
                raise ValueError(f"{_format_path(path)} must be an object schema")

            cache_key = f"{current_model_name}:{id(object_schema)}"
            cached = model_cache.get(cache_key)
            if cached is not None:
                return cached

            properties = normalized_schema.get("properties", {})
            required = normalized_schema.get("required", [])
            additional_properties = normalized_schema.get("additionalProperties", True)

            if not isinstance(required, list):
                raise ValueError(f"{_format_path(path)} 'required' must be an array")

            if not isinstance(properties, dict):
                raise ValueError(f"{_format_path(path)} 'properties' must be an object")

            if not properties:
                if isinstance(additional_properties, dict):
                    raise ValueError(
                        f"{_format_path(path)} is a pure additionalProperties map and should "
                        "be used as a field type, not as a model root"
                    )
                raise ValueError(
                    f"{_format_path(path)} must have a non-empty 'properties' object"
                )

            field_definitions: dict[str, Any] = {}
            for field_name, field_schema in properties.items():
                child_path = f"{path}.{field_name}" if path else field_name
                annotation = _schema_to_type(field_schema, child_path)
                if field_name in required:
                    field_definitions[field_name] = (annotation, ...)
                else:
                    field_definitions[field_name] = (annotation | None, None)

            if additional_properties not in (True, False):
                raise ValueError(
                    f"{_format_path(path)} only supports boolean additionalProperties "
                    "when combined with named properties"
                )

            model_config = ConfigDict(
                extra="forbid" if additional_properties is False else "ignore",
                use_enum_values=True,
            )
            dynamic_model = create_model(
                current_model_name, __config__=model_config, **field_definitions
            )
            model_cache[cache_key] = dynamic_model
            return dynamic_model
        finally:
            if current_ref_name is not None:
                in_flight_refs.remove(current_ref_name)

    if not isinstance(schema, dict):
        raise ValueError("Schema must be a dictionary")

    return _object_schema_to_model(schema, _sanitize_model_name(model_name), "")
