---
title: "Legacy Code Integration with @mify"
description: "Apply the @mify decorator to existing Python classes so a Mellea session can act on, query, and transform your objects without rewriting them."
# diataxis: reference
---

This example shows how to bring existing Python objects into a Mellea session
using the `@mify` decorator. `@mify` adds the `MifiedProtocol` interface to a
class or instance so you can pass it directly to session methods like `m.act()`,
`m.query()`, and `m.transform()`.

**Source file:** `docs/examples/mify/mify.py`

## Concepts covered

- Applying `@mify` as a class decorator
- Mifying an object instance at runtime (ad-hoc mification)
- Controlling string representation with `stringify_func`
- Choosing a query template with `query_type` and `template_order`
- Selecting which fields the model sees with `fields_include`
- Exposing specific methods as tools with `funcs_include`

## Prerequisites

- [Quick Start](../getting-started/quickstart) complete
- [MObjects and mify](../concepts/mobjects-and-mify) concept page (recommended background)
- Ollama running locally with `granite4.1:3b` pulled

## The full example

### Imports

```python
from mellea.stdlib.components.docs.richdocument import TableQuery
from mellea.stdlib.components.mify import MifiedProtocol, mify
from mellea.stdlib.session import start_session
```

`MifiedProtocol` is used here only for the `isinstance` assertion that
demonstrates what `@mify` adds to a class. In production code you would not
normally need to import it.

### Mifying a class with the decorator

```python
# Mify works on python objects and classes. Apply it to your own
# custom class or object to start working with mellea.
@mify
class MyCustomerClass:
    def __init__(self, name: str, last_purchase: str) -> None:
        self.name = name
        self.last_purchase = last_purchase


# Now when you instantiate an object of that class, it will also
# have the fields and members necessary for working with mellea.
c = MyCustomerClass("Jack", "Beans")
assert isinstance(c, MifiedProtocol)
```

Applying `@mify` to a class is a one-liner. Every instance of the decorated
class automatically satisfies `MifiedProtocol`, which means you can pass any
instance to a session method without any further setup.

### Ad-hoc mification of an existing instance

```python
# You can also mify objects ad hoc.
class MyStoreClass:
    def __init__(self, purchases: list[str]) -> None:
        self.purchases: list[str]


store = MyStoreClass(["Beans", "Soil", "Watering Can"])
mify(store)
assert isinstance(store, MifiedProtocol)

# Now, you can use these objects in MelleaSessions.
store.format_for_llm()
m = start_session()
m.act(store)
```

You do not have to own a class to mify it. Call `mify(instance)` on any object
to patch in the protocol at runtime. This is useful when integrating with
third-party libraries or legacy code you cannot modify.

Note that `m.act(store)` without a custom string representation will not produce
useful output unless the class defines `__str__`. The next section shows how to
supply one.

### Custom string representation

```python
# However, unless your object/class has a __str__ function,
# this won't do much good by itself. You need to specify how
# mellea should process these objects as text. You can do this by
# parameterizing mify.
@mify(stringify_func=lambda x: f"Chain Location: {x.location}")  # type: ignore
class MyChain:
    def __init__(self, location: str):
        self.location = location


# M operations will now utilize that string representation of the
# object when interacting with it.
m.query(MyChain("Northeast"), "Where is my chain located?")
```

`stringify_func` accepts a callable that takes the instance and returns a
string. The lambda here produces a short, labelled description. Any callable
works — a method on another object, a formatting helper, or a template
renderer.

### Template integration with TableQuery

```python
# For more complicated representations, you can utilize mify
# to interact with our templating system. Here, we know that a
# TableQuery calls its underlying object's to_markdown function.
# Since our class has the same process, we can use that template.
# We can also specify that our class should use either a template with it's own
# class name or the Table template when not querying.
@mify(query_type=TableQuery, template_order=["*", "Table"])
class MyCompanyDatabase:
    table: str = """| Store      | Sales   |
| ---------- | ------- |
| Northeast  | $250    |
| Southeast  | $80     |
| Midwest    | $420    |"""

    def to_markdown(self):
        return self.table
```

`query_type=TableQuery` tells Mellea which query component to use when
`m.query()` is called on this object. `template_order` controls the fallback
chain for rendering: try the class-specific template first (`"*"`), then fall
back to the generic `"Table"` template.

### Field selection and inline templates

```python
# Mellea also allows you to specify the fields you want to
# include from your class and a corresponding template that
# takes those fields.
@mify(fields_include={"table"}, template="{{ table }}")
class MyOtherCompanyDatabase:
    table: str = """| Store      | Sales   |
| ---------- | ------- |
| Northeast  | $250    |
| Southeast  | $80     |
| Midwest    | $420    |"""


m.query(
    MyOtherCompanyDatabase(), "What were sales for the Northeast branch this month?"
)
```

`fields_include` limits which attributes are visible to the model. Sensitive or
irrelevant fields stay private. The `template` parameter is a Jinja2 string
rendered with the included fields as context variables.

### Exposing methods as tools

```python
# By default, mifying and object will also provide any functions
# of your class/object to models as tools in m functions that support tools.
# The default behavior only includes functions that have docstrings without
# [no-index] in it.
@mify(funcs_include={"from_markdown"})
class MyDocumentLoader:
    def __init__(self) -> None:
        self.content = ""

    @classmethod
    def from_markdown(cls, text: str) -> "MyDocumentLoader":
        doc = MyDocumentLoader()
        # Your parsing functions here.
        doc.content = text
        return doc


# m.transform will be able to call the from_markdown function to return
# the poem as a MyDocumentLoader object.
m.transform(MyDocumentLoader(), "Write a poem.")
```

`funcs_include` whitelists specific methods. The model can call `from_markdown`
as a tool when `m.transform()` runs. By default, any method that has a
docstring (and whose docstring does not contain `[no-index]`) is exposed.
`funcs_include` overrides that default to give you precise control.

### Full file

```python
# pytest: ollama, llm

from mellea.stdlib.components.docs.richdocument import TableQuery
from mellea.stdlib.components.mify import MifiedProtocol, mify
from mellea.stdlib.session import start_session


# Mify works on python objects and classes. Apply it to your own
# custom class or object to start working with mellea.
@mify
class MyCustomerClass:
    def __init__(self, name: str, last_purchase: str) -> None:
        self.name = name
        self.last_purchase = last_purchase


# Now when you instantiate an object of that class, it will also
# have the fields and members necessary for working with mellea.
c = MyCustomerClass("Jack", "Beans")
assert isinstance(c, MifiedProtocol)


# You can also mify objects ad hoc.
class MyStoreClass:
    def __init__(self, purchases: list[str]) -> None:
        self.purchases: list[str]


store = MyStoreClass(["Beans", "Soil", "Watering Can"])
mify(store)
assert isinstance(store, MifiedProtocol)

# Now, you can use these objects in MelleaSessions.
store.format_for_llm()
m = start_session()
m.act(store)


# However, unless your object/class has a __str__ function,
# this won't do much good by itself. You need to specify how
# mellea should process these objects as text. You can do this by
# parameterizing mify.
@mify(stringify_func=lambda x: f"Chain Location: {x.location}")  # type: ignore
class MyChain:
    def __init__(self, location: str):
        self.location = location


# M operations will now utilize that string representation of the
# object when interacting with it.
m.query(MyChain("Northeast"), "Where is my chain located?")


# For more complicated representations, you can utilize mify
# to interact with our templating system. Here, we know that a
# TableQuery calls its underlying object's to_markdown function.
# Since our class has the same process, we can use that template.
# We can also specify that our class should use either a template with it's own
# class name or the Table template when not querying.
@mify(query_type=TableQuery, template_order=["*", "Table"])
class MyCompanyDatabase:
    table: str = """| Store      | Sales   |
| ---------- | ------- |
| Northeast  | $250    |
| Southeast  | $80     |
| Midwest    | $420    |"""

    def to_markdown(self):
        return self.table


# Mellea also allows you to specify the fields you want to
# include from your class and a corresponding template that
# takes those fields.
@mify(fields_include={"table"}, template="{{ table }}")
class MyOtherCompanyDatabase:
    table: str = """| Store      | Sales   |
| ---------- | ------- |
| Northeast  | $250    |
| Southeast  | $80     |
| Midwest    | $420    |"""


m.query(
    MyOtherCompanyDatabase(), "What were sales for the Northeast branch this month?"
)


# By default, mifying and object will also provide any functions
# of your class/object to models as tools in m functions that support tools.
# The default behavior only includes functions that have docstrings without
# [no-index] in it.
@mify(funcs_include={"from_markdown"})
class MyDocumentLoader:
    def __init__(self) -> None:
        self.content = ""

    @classmethod
    def from_markdown(cls, text: str) -> "MyDocumentLoader":
        doc = MyDocumentLoader()
        # Your parsing functions here.
        doc.content = text
        return doc


# m.transform will be able to call the from_markdown function to return
# the poem as a MyDocumentLoader object.
m.transform(MyDocumentLoader(), "Write a poem.")
```

## Key observations

**`@mify` is additive.** It does not subclass, wrap, or monkey-patch the class
in a destructive way. Existing behaviour is unchanged; the protocol members are
added on top.

**Ad-hoc mification is instance-scoped.** Calling `mify(instance)` mutates
only that object. Other instances of the same class are not affected.

**`fields_include` is the privacy boundary.** If your class holds credentials,
internal state, or large fields you do not want sent to the model, list only the
fields the model should see.

**Tool exposure is opt-in by default.** Only methods with non-empty docstrings
(without `[no-index]`) are exposed as tools. Use `funcs_include` to be
explicit.

**What to try next:**

- Read the [MObjects and mify](../concepts/mobjects-and-mify) concept page for
  the full design rationale.
- See `docs/examples/mify/rich_document_advanced.py` for mify combined with
  rich document types.
- See `docs/examples/mify/rich_table_execute_basic.py` for mifying table
  objects for data manipulation.

---

**See also:** [MObjects and mify](../concepts/mobjects-and-mify) | [Tutorial 05: MIFYing Legacy Code](../tutorials/05-mifying-legacy-code) | [Examples Index](./index)
