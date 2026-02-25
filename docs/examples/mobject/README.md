# MObject Examples

This directory contains examples of using MObjects (Mellea Objects) - structured data types that work seamlessly with LLMs.

## Files

### table.py
Demonstrates using table MObjects for structured data manipulation.

**Key Features:**
- Creating and querying tables
- Table transformations
- Structured data operations
- Type-safe table manipulation

## Concepts Demonstrated

- **Structured Data**: Working with tables and structured formats
- **Query Operations**: Asking questions about data
- **Transformations**: Modifying data based on instructions
- **Type Safety**: Maintaining structure through operations
- **LLM Integration**: Using LLMs to work with structured data

## Basic Usage

```python
from mellea.stdlib.components import Table
from mellea import start_session

# Create a table
table = Table(data=[
    {"name": "Alice", "age": 30, "city": "NYC"},
    {"name": "Bob", "age": 25, "city": "SF"},
])

m = start_session()

# Query the table
result = m.query(table, "What is the average age?")

# Transform the table
new_table = m.transform(table, "Add a column for country (USA)")
```

## MObject Types

Mellea provides several MObject types:
- **Table**: Tabular data with rows and columns
- **Document**: Text documents with metadata
- **RichDocument**: Documents with rich formatting
- Custom MObjects via `@mify`

## Operations

### Query
Ask questions about the data without modifying it:
```python
answer = m.query(mobject, "What is the total?")
```

### Transform
Modify the data based on instructions:
```python
new_mobject = m.transform(mobject, "Sort by date descending")
```

### Execute
Perform operations that may have side effects:
```python
result = m.execute(mobject, "Calculate statistics")
```

## Related Documentation

- See `mellea/stdlib/components/mobject.py` for MObject base class
- See `mellea/stdlib/components/docs/` for document MObjects
- See `mify/` for creating custom MObjects
- See `notebooks/table_mobject.ipynb` for interactive examples