# Tool Argument Validation Test Suite

This test suite provides comprehensive coverage for tool call argument validation scenarios.

## Purpose

These tests document the **current behavior** and **expected behavior** of tool argument handling across Mellea backends. Many tests currently **fail** or demonstrate issues that the proposed validation system will address.

## Test Categories

### 1. Basic Type Validation (`TestBasicTypeValidation`)
- String, integer, float, boolean arguments
- Tests that these work when types match

### 2. Type Coercion (`TestTypeCoercion`)
- String to int: `"30"` → `30`
- String to float: `"95.5"` → `95.5`
- Int to string: `123` → `"123"`
- String to bool: `"true"` → `True`

**Current Status:** Most coercion tests FAIL without validation

### 3. Optional Parameters (`TestOptionalParameters`)
- Optional parameters provided, omitted, or set to None
- Default values (all provided, partial, minimal)

### 4. Union Types (`TestUnionTypes`)
- `Union[str, int]` parameters
- Tests both string and integer values

### 5. Complex Types (`TestComplexTypes`)
- Lists: `list[str]`, empty lists
- Dicts: `dict[str, Any]`
- Nested structures: `dict[str, list[int]]`

### 6. Error Conditions (`TestErrorConditions`)
- Missing required arguments
- Extra arguments
- Wrong types without coercion
- None for required parameters

**Current Status:** These demonstrate current failure modes

### 7. JSON Parsing (`TestJSONParsing`)
- Valid JSON strings
- Malformed JSON
- Escaped quotes, unicode, nested objects

### 8. Pydantic Models (`TestPydanticModels`)
- Tools accepting Pydantic model arguments
- Validation errors
- Optional fields

### 9. Edge Cases (`TestEdgeCases`)
- No-parameter tools
- Hallucinated arguments
- Very long strings
- Special characters
- Empty strings

### 10. parse_tools() Function (`TestParseToolsFunction`)
- Parsing tool calls from text (HuggingFace backend)
- Single and multiple tool calls
- Malformed JSON handling

## Running the Tests

### Run all tests:
```bash
uv run pytest test/backends/test_tool_argument_validation.py -v
```

### Run specific test class:
```bash
uv run pytest test/backends/test_tool_argument_validation.py::TestBasicTypeValidation -v
```

### Run specific test:
```bash
uv run pytest test/backends/test_tool_argument_validation.py::TestTypeCoercion::test_string_to_int_coercion -v
```

### Run with markers:
```bash
# Skip integration tests (not yet implemented)
uv run pytest test/backends/test_tool_argument_validation.py -v -m "not integration"
```

## Expected Results

### Currently Passing Tests
- Basic type validation when types match
- Optional parameters
- Complex types (lists, dicts) when properly formatted
- JSON parsing with valid input

### Currently Failing Tests
These tests **document issues** that validation will fix:

1. **Type Coercion Tests** - Fail because no automatic coercion
   - `test_string_to_int_coercion`
   - `test_string_to_float_coercion`
   - `test_string_to_bool_coercion`

2. **Error Condition Tests** - Fail at call time, not validation time
   - `test_missing_required_argument`
   - `test_extra_arguments`
   - `test_wrong_type_no_coercion`

3. **Edge Case Tests**
   - `test_no_parameters_with_hallucinated_args` - Models hallucinate params

### Integration Tests
Marked with `@pytest.mark.skip` - will be enabled after validation implementation:
- `test_validation_with_coercion`
- `test_validation_strict_mode`

## Test Data

The test file includes realistic tool function signatures:

```python
def typed_primitives_tool(name: str, age: int, score: float, active: bool) -> dict
def optional_params_tool(required: str, optional: Optional[str] = None) -> str
def union_type_tool(value: Union[str, int]) -> str
def list_param_tool(items: list[str]) -> int
def dict_param_tool(config: dict[str, Any]) -> str
def nested_structure_tool(data: dict[str, list[int]]) -> int
```

## Adding New Tests

When adding tests:

1. **Document the scenario** in the docstring
2. **Mark expected failures** with `pytest.mark.xfail` if appropriate
3. **Add to relevant test class** or create new class
4. **Include both success and failure cases**

Example:
```python
def test_new_scenario(self):
    """Test description of what this validates."""
    args = {"param": "value"}
    tool_call = ModelToolCall("tool_name", tool_func, args)
    result = tool_call.call_func()
    assert result == expected_value
```

## Integration with Validation System

Once the validation system is implemented:

1. **Update integration tests** - Remove `@pytest.mark.skip`
2. **Add validation calls** - Test with `strict=True` and `strict=False`
3. **Verify coercion** - Ensure type coercion tests pass
4. **Test error messages** - Validate helpful error messages

Example with validation:
```python
from mellea.backends.tools import validate_tool_arguments

def test_with_validation(self):
    args = {"name": "Test", "age": "30"}  # age is string
    validated_args = validate_tool_arguments(func, args, coerce_types=True)
    assert validated_args["age"] == 30  # Coerced to int
    assert isinstance(validated_args["age"], int)
```

## Related Documentation

- [Tool Call Argument Validation Investigation](../../docs/dev/tool_call_argument_validation_investigation.md)
- [Tool Calling Documentation](../../docs/dev/tool_calling.md)
- [Backend Tools Module](../../mellea/backends/tools.py)

## Notes

- Tests use `ModelToolCall` directly to isolate argument handling from backend-specific code
- Some tests intentionally trigger errors to document current behavior
- Complex type tests cover real-world LLM output scenarios
- JSON parsing tests cover common LLM formatting issues