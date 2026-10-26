# Information Extraction Examples

Examples demonstrating structured information extraction from text using Mellea.

## Examples

| File | Description |
| :--- | :--- |
| [`101_with_gen_stubs.py`](./101_with_gen_stubs.py) | Declarative named entity extraction using a `@generative` function with a `list[str]` return type. |
| [`advanced_with_m_instruct.py`](./advanced_with_m_instruct.py) | Constrained extraction using `m.instruct()` with a Pydantic `format` model, a custom predicate requirement, and `RejectionSamplingStrategy`. |
