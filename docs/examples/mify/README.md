# Mify Examples

This directory demonstrates the `@mify` decorator for making custom Python objects work seamlessly with Mellea.

## Files

### mify.py
Comprehensive examples of using `@mify` to integrate custom classes with Mellea.

**Key Features:**
- Basic class mification with `@mify` decorator
- Ad-hoc object mification
- Custom string representations with `stringify_func`
- Template integration with `query_type` and `template_order`
- Field selection with `fields_include`
- Function exposure as tools with `funcs_include`

### rich_document_advanced.py
Advanced examples using mify with rich document types.

### rich_table_execute_basic.py
Examples of mifying table objects for data manipulation.

## Concepts Demonstrated

- **Object Integration**: Making custom objects LLM-compatible
- **Template System**: Using Jinja2 templates for object rendering
- **Tool Generation**: Automatically exposing methods as LLM tools
- **Type Safety**: Maintaining Python type information
- **Composability**: Using mified objects in Mellea operations

## Basic Usage

```python
from mellea.stdlib.components.mify import mify
from mellea import start_session

# Mify a class
@mify
class Customer:
    def __init__(self, name: str, last_purchase: str):
        self.name = name
        self.last_purchase = last_purchase

# Use in Mellea
customer = Customer("Jack", "Beans")
m = start_session()
m.act(customer)
```

## Advanced Usage

```python
# Custom string representation
@mify(stringify_func=lambda x: f"Location: {x.location}")
class Store:
    def __init__(self, location: str):
        self.location = location

# Template integration
@mify(query_type=TableQuery, template_order=["*", "Table"])
class Database:
    def to_markdown(self):
        return self.table_data

# Field selection
@mify(fields_include={"name", "value"}, template="{{ name }}: {{ value }}")
class Config:
    name: str
    value: str
    internal_data: dict  # Not included

# Function exposure
@mify(funcs_include={"process", "validate"})
class Processor:
    def process(self, data: str) -> str:
        """Process the data."""
        return data.upper()
    
    def validate(self, data: str) -> bool:
        """Validate the data."""
        return len(data) > 0
```

## MifiedProtocol

Objects decorated with `@mify` implement the `MifiedProtocol`, which provides:
- `format_for_llm()`: Format object for LLM consumption
- Template rendering support
- Tool function exposure
- Integration with Mellea operations

## Related Documentation

- See `mellea/stdlib/components/mify.py` for implementation
- See `docs/dev/mify.md` for design details
- See `mellea/templates/` for template system