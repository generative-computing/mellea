# Helper Utilities

This directory contains helper functions and utilities used across multiple examples.

## Files

| File | Description |
| :--- | :--- |
| [`helpers.py`](./helpers.py) | Common utility functions used by example scripts. Key functions: `w()`, a text wrapping function for pretty-printing long outputs, and other formatting and display utilities. |

## Usage

```python
from docs.examples.helper import w

# Wrap long text for display
print(w(long_text))
```

## Note

This is a utility directory to support other examples. The helpers are not part of the core Mellea API but are useful for demonstration purposes.
