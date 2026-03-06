---
title: "Template formatting"
description: "How Mellea's TemplateFormatter converts Python objects into model-ready text using Jinja2 templates."
# diataxis: explanation
---

# Template formatting

Most backends operate on text. Mellea converts Python objects to text using the
`TemplateFormatter` — a Jinja2-based system that lets you control exactly how each component
type is rendered for the model.

This page is for contributors and advanced users who need to customize how objects are
represented in prompts.

## Templates

The `TemplateFormatter` uses Jinja2 templates stored in a directory tree under
`mellea/templates/prompts/`. Each component type has a corresponding `.jinja2` file that
controls its textual representation. The default templates are in
`mellea/templates/prompts/default/`.

Templates can also be stored directly on the class by returning a `TemplateRepresentation`
from `format_for_llm()`, rather than relying on a directory lookup.

## Template lookup order

When rendering a component, the `TemplateFormatter` searches for a matching template in this
order:

1. The formatter's in-memory cache (if the template has been looked up recently)
2. The formatter's configured template path
3. The package that owns the object being formatted (`mellea` or a third-party package)

When searching a directory, the formatter traverses subdirectories that match the current
model ID — for example, `ibm-granite/granite-3.2-8b-instruct` matches:

```text
templates/prompts/granite/granite-3-2/instruct/
```

or falls back to:

```text
templates/prompts/default/
```

The deepest matching directory wins. A given `templates/` directory should not contain
multiple matches for the same model ID (e.g. both `granite/` and `ibm/` paths for the same
model string).

## Template representations

A component's `format_for_llm()` method controls how it is rendered. It returns either a
plain string or a `TemplateRepresentation` object.

**Plain string** — skip the template engine entirely:

```python
def format_for_llm(self) -> str:
    return f"Table with {len(self.rows)} rows:\n{self.to_markdown()}"
```

**`TemplateRepresentation`** — use the template engine:

```python
from mellea.stdlib.components import TemplateRepresentation

def format_for_llm(self) -> TemplateRepresentation:
    return TemplateRepresentation(
        component=self,
        args={"table": self.to_markdown(), "title": self.title},
        tools=[],
        template_order=["my_component", "*"],  # * = class name
    )
```

`TemplateRepresentation` fields:

| Field | Description |
| ----- | ----------- |
| `component` | The object being rendered (usually `self`) |
| `args` | Dict of variables passed to the Jinja2 template |
| `tools` | List of tool/function descriptors exposed to the model |
| `template` | Inline Jinja2 template string (alternative to `template_order`) |
| `template_order` | List of template filenames to search for, in priority order |

## Customizing templates for a component

To customize how an existing component is formatted for a specific model, subclass it and
override `format_for_llm()`, then create a new `.jinja2` template file.

```python
class MyCustomTable(Table):
    def format_for_llm(self) -> TemplateRepresentation:
        return TemplateRepresentation(
            component=self,
            args={"table": self.to_markdown()},
            tools=list(self._get_tools()),
            template_order=["my_custom_table", "table", "*"],
        )
```

Place the template file at:

```text
your_package/templates/prompts/default/my_custom_table.jinja2
```

or at a model-specific path:

```text
your_package/templates/prompts/granite/granite-3-2/instruct/my_custom_table.jinja2
```

The model-specific template will be used for that model; all others fall back to `default/`.

> **Advanced:** For a worked example of advanced template customization, see
> [`docs/examples/mify/rich_document_advanced.py`](https://github.com/generative-computing/mellea/blob/main/docs/examples/mify/rich_document_advanced.py)
> in the source repository.

**See also:** [MObjects and mify](../concepts/mobjects-and-mify) |
[Mellea core internals](./mellea-core-internals)
