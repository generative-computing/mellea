---
title: "MObjects and mify"
description: "How the @mify decorator turns any Python class into an LLM-queryable object with controlled field and method exposure."
# diataxis: explanation
---

# MObjects and mify

Object-oriented programming organizes related data and the methods that operate on it into
classes. Mellea applies the same principle to LLM interactions: an **MObject** is a Python
class whose fields and methods can be exposed to a model in a controlled, structured way.

The `@mify` decorator turns any class into an MObject. You specify exactly which fields and
methods are visible to the LLM — nothing else is exposed.

## The @mify decorator

```python
import mellea
from mellea.stdlib.components import mify
from mellea.stdlib.components.mify import MifiedProtocol

@mify(fields_include={"table"}, template="{{ table }}")
class SalesDatabase:
    table: str = """| Store      | Sales  |
                    | ---------- | ------ |
                    | Northeast  | $250   |
                    | Southeast  | $80    |
                    | Midwest    | $420   |"""

    def internal_method(self):
        # not exposed to the LLM
        ...

m = mellea.start_session()
db = SalesDatabase()
assert isinstance(db, MifiedProtocol)

answer = m.query(db, "What were sales for the Northeast branch this month?")
print(str(answer))
# Output will vary — LLM responses depend on model and temperature.
```

`fields_include` controls which fields appear in the prompt. `template` is a Jinja2 template
that controls how those fields are rendered. The `m.query()` call sends the rendered object
plus the question to the model.

`@mify` is useful whenever you need to expose structured data to a model without leaking
internal state.

## Methods as tools

When you `mify` a class, every method that has a docstring is automatically registered as a
tool the LLM can call. Use `funcs_include` or `funcs_exclude` to control which methods
are exposed:

```python
from mellea.stdlib.components import mify

@mify(funcs_include={"from_markdown"})
class DocumentLoader:
    def __init__(self) -> None:
        self.content = ""

    @classmethod
    def from_markdown(cls, text: str) -> "DocumentLoader":
        """Load a document from a Markdown string."""
        doc = DocumentLoader()
        doc.content = text
        return doc

    def internal_helper(self) -> str:
        # no docstring, and not in funcs_include — never exposed
        return "..."
```

Only `from_markdown` is registered as a tool. The model can call it during a `m.transform()`
or `m.query()` operation; `internal_helper` is invisible.

When a class method and an LLM operation would produce the same result, Mellea will note that
the direct method call is available:

```python
# Both of these transform the table in the same way.
# Mellea will suggest using the direct method call instead.
table_transposed = m.transform(table, "Transpose the table.")
table_transposed_direct = table.transpose()
```

## Working with documents

Mellea provides `mified` wrappers around [Docling](https://github.com/docling-project/docling)
documents for working with PDFs and other rich documents.

```python
from mellea.stdlib.docs.richdocument import RichDocument

rd = RichDocument.from_document_file("https://arxiv.org/pdf/1906.04043")
```

This loads the PDF and parses it into Mellea's intermediate representation. From there you can
extract structured elements:

```python
from mellea.stdlib.docs.richdocument import Table

table: Table = rd.get_tables()[0]
print(table.to_markdown())
```

`Table` is already an MObject, so you can pass it directly to `m.transform()` or `m.query()`:

```python
from mellea.backends import ModelOption
from mellea import start_session

m = start_session()

# Try a few seeds to find a run that returns a parsable table
for seed in [x * 12 for x in range(5)]:
    result = m.transform(
        table,
        "Add a column 'Model' that extracts which model was used, or 'None' if none.",
        model_options={ModelOption.SEED: seed},
    )
    if isinstance(result, Table):
        print(result.to_markdown())
        break
```

The seed loop is a simple retry strategy: LLM output is non-deterministic, so iterating
over seeds gives multiple independent samples until one produces a valid table structure.

> **Note:** LLM output is non-deterministic. Your exact results will vary.

## When to use MObjects

MObjects are well-suited for:

- **Document querying** — wrap a document, expose only the relevant sections, query or
  transform them with the model
- **Tool registration** — expose a controlled set of methods as tools the LLM can invoke
  during generation
- **Evolving existing codebases** — add `@mify` to an existing class to make it
  LLM-accessible without rewriting it

For simple one-off generation, `m.instruct()` is usually sufficient. MObjects add value when
you have structured data or methods that the model needs to reason about or call.

**See also:** [Context and Sessions](./context-and-sessions) |
[Generative Functions](./generative-functions)
