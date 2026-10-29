# Mini Researcher Example

This directory contains a complete example application: a mini research assistant that uses RAG (Retrieval-Augmented Generation) with validation.

## Files

| File | Description |
| :--- | :--- |
| [`researcher.py`](./researcher.py) | Main implementation of the research assistant with helper functions. |
| [`context_docs.py`](./context_docs.py) | Document context and RAG document definitions. |
| [`__init__.py`](./__init__.py) | Package initialization and exports. |

## Architecture

```
User Query
    ↓
Document Retrieval (RAG)
    ↓
Generation with Context
    ↓
Validation:
  - Word count check
  - Citation validation
  - Safety check (Guardian)
    ↓
Result
```
