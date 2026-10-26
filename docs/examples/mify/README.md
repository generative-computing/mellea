# Mify Examples

Examples demonstrating the `@mify` decorator for making custom Python objects
LLM-compatible with Mellea. For a conceptual overview, see
[MObjects and mify](../../docs/docs/concepts/mobjects-and-mify.md).

## Examples

| File | Description |
| :--- | :--- |
| [`mify.py`](./mify.py) | Core `@mify` patterns: class decoration, ad-hoc object mification, `stringify_func`, `query_type` and `template_order` for template integration, `fields_include` with inline templates, and `funcs_include` for tool exposure. |
| [`rich_document_advanced.py`](./rich_document_advanced.py) | Progressive walkthrough of PDF document processing: raw markdown export, `RichDocument` wrapper, a custom `format_for_llm()` subclass with an inline Jinja2 template, and grounded generation. Requires `docling`. |
| [`rich_table_execute_basic.py`](./rich_table_execute_basic.py) | Extracting a mified `Table` from a PDF with `RichDocument` and applying `m.transform()` with seed-based retries to add a derived column. Requires `docling`. |
