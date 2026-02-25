# Tutorial Examples

This directory contains Python script versions of the tutorial notebooks, suitable for running directly or studying the code.

## Files

### example.py
General introduction to Mellea with basic examples.

### simple_email.py
Email generation example - the classic "Hello World" of Mellea.

### instruct_validate_repair.py
Walkthrough of the instruct-validate-repair paradigm.

### compositionality_with_generative_slots.py
Tutorial on composing generative functions.

### context_example.py
Working with contexts and context management.

### document_mobject.py
Using document MObjects for text processing.

### table_mobject.py
Working with table data structures.

### model_options_example.py
Configuring model options and parameters.

### sentiment_classifier.py
Building a sentiment classification system.

### mcp_example.py
Model Context Protocol integration examples.

## Relationship to Notebooks

Each file in this directory corresponds to a Jupyter notebook in `notebooks/`:
- `example.py` ↔ `example.ipynb`
- `simple_email.py` ↔ `simple_email.ipynb`
- `instruct_validate_repair.py` ↔ `instruct_validate_repair.ipynb`
- etc.

## Running the Examples

```bash
# Run any example directly
python docs/examples/tutorial/simple_email.py

# Or with uv
uv run docs/examples/tutorial/simple_email.py
```

## Learning Path

Recommended order for learning Mellea:

1. **simple_email.py** - Start here for basic concepts
2. **instruct_validate_repair.py** - Core paradigm
3. **compositionality_with_generative_slots.py** - Building blocks
4. **context_example.py** - Context management
5. **model_options_example.py** - Configuration
6. **sentiment_classifier.py** - Complete application
7. **document_mobject.py** - Working with documents
8. **table_mobject.py** - Working with structured data
9. **mcp_example.py** - Advanced integration

## Key Concepts Covered

- **Sessions**: Creating and managing Mellea sessions
- **Instructions**: Generating text with natural language instructions
- **Requirements**: Constraining outputs with validation
- **Validation**: Checking outputs against requirements
- **Repair**: Automatically fixing invalid outputs
- **Generative Slots**: Type-safe LLM functions
- **Contexts**: Managing conversation and generation history
- **Model Options**: Configuring model behavior
- **MObjects**: Working with structured data types

## Differences from Notebooks

- **No Interactive Output**: Scripts don't show intermediate results
- **Linear Execution**: Runs from top to bottom
- **Easier to Version Control**: Plain Python files
- **Better for CI/CD**: Can be run in automated pipelines
- **Easier to Import**: Can import functions from these files

## Converting to Notebooks

To convert a script to a notebook:
```bash
# Install jupytext
uv pip install jupytext

# Convert
jupytext --to notebook docs/examples/tutorial/simple_email.py
```

## Related Documentation

- See `notebooks/` for interactive Jupyter versions
- See other example directories for specialized topics
- See main README.md for getting started guide