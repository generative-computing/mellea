# MObject Examples

Examples demonstrating the MObject pattern — using `@mify` to make structured Python objects
queryable and transformable by Mellea sessions. For a full explanation and more patterns, see
[MObjects and mify](../../docs/concepts/mobjects-and-mify.md) and
[Working with Data](../../docs/how-to/working-with-data.md).

## Examples

| File | Description |
| :--- | :--- |
| [`table.py`](./table.py) | Querying and updating a markdown table via `m.query()` and `m.transform()`, using `@mify` with `fields_include` and a Jinja2 template. An interactive Colab version is in [`notebooks/table_mobject.ipynb`](../notebooks/table_mobject.ipynb). |
