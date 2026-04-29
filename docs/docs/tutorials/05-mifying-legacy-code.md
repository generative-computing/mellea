---
title: "Tutorial: Mifying Legacy Code"
description: "Add LLM query and transform capabilities to existing Python classes without rewriting them."
# diataxis: tutorial
---

> **Concept overview:** [MObjects and mify](../concepts/mobjects-and-mify) explains the design and trade-offs.

This tutorial shows how to make existing Python objects queryable and transformable
by the LLM using [`@mify`](../reference/glossary#mify--mify) — without changing their Python interface or behaviour.

By the end you will have covered:

- Applying `@mify` to an existing class
- `m.query()` — ask questions about an object
- `m.transform()` — produce a transformed version of an object
- Controlling which fields and methods the LLM sees
- Using `stringify_func` for custom text representations

**Prerequisites:** [Tutorial 01](./01-your-first-generative-program) complete,
`pip install mellea`, Ollama running locally with `granite4.1:3b` downloaded.

---

## The scenario

You have a `CustomerRecord` class — existing code that you cannot rewrite. You want
to start asking the LLM questions about individual records and generating
personalised summaries.

```python
class CustomerRecord:
    def __init__(self, name: str, last_purchase: str, spend_ytd: float):
        self.name = name
        self.last_purchase = last_purchase
        self.spend_ytd = spend_ytd
```

## Step 1: Apply @mify

Decorate the class with `@mify`. This adds the LLM-queryable protocol to every
instance, without touching the class's Python interface:

```python
# Requires: mellea
# Returns: str
import mellea
from mellea.stdlib.components.mify import mify

@mify
class CustomerRecord:
    def __init__(self, name: str, last_purchase: str, spend_ytd: float):
        self.name = name
        self.last_purchase = last_purchase
        self.spend_ytd = spend_ytd

record = CustomerRecord("Ada", "wireless headphones", 1240.50)

m = mellea.start_session()
result = m.query(record, "What was this customer's last purchase?")
print(str(result))
# Output will vary — LLM responses depend on model and temperature.
```

By default, `@mify` exposes all instance attributes as fields and adds the
[`MObject`](../reference/glossary#mobject) protocol to every instance. The LLM sees a text representation
of the object built from those fields.

> **Full example:** [`docs/examples/mify/mify.py`](https://github.com/generative-computing/mellea/blob/main/docs/examples/mify/mify.py)

## Step 2: Control the text representation

If the default field listing is too verbose or structured incorrectly, supply a
`stringify_func` to produce exactly the text the LLM receives:

```python
# Requires: mellea
# Returns: str
import mellea
from mellea.stdlib.components.mify import mify

@mify(stringify_func=lambda r: (
    f"Customer: {r.name}\n"
    f"Last purchase: {r.last_purchase}\n"
    f"Year-to-date spend: £{r.spend_ytd:.2f}"
))
class CustomerRecord:
    def __init__(self, name: str, last_purchase: str, spend_ytd: float):
        self.name = name
        self.last_purchase = last_purchase
        self.spend_ytd = spend_ytd

record = CustomerRecord("Ada", "wireless headphones", 1240.50)
m = mellea.start_session()

result = m.query(record, "Is this a high-value customer?")
print(str(result))
# Output will vary — LLM responses depend on model and temperature.
```

## Step 3: Limit which fields are visible

To hide internal state from the LLM, use `fields_include` with a Jinja2 template:

```python
# Requires: mellea
# Returns: str
import mellea
from mellea.stdlib.components.mify import mify

@mify(
    fields_include={"name", "spend_ytd"},
    template="{{ name }} — spent £{{ spend_ytd }} this year",
)
class CustomerRecord:
    def __init__(self, name: str, last_purchase: str, spend_ytd: float):
        self.name = name
        self.last_purchase = last_purchase
        self.spend_ytd = spend_ytd

record = CustomerRecord("Ada", "wireless headphones", 1240.50)
m = mellea.start_session()

result = m.query(record, "Classify this customer as low, medium, or high value.")
print(str(result))
# Output will vary — LLM responses depend on model and temperature.
```

The `last_purchase` field is not in `fields_include` so it is never sent to the
model.

## Step 4: Use m.transform()

`m.transform()` asks the LLM to produce a modified version of the object by
calling one of its methods. Expose the target method with `funcs_include`:

```python
# Requires: mellea
# Returns: str
import mellea
from mellea.stdlib.components.mify import mify

@mify(
    stringify_func=lambda r: f"{r.name}: {r.last_purchase}, £{r.spend_ytd:.2f} YTD",
    funcs_include={"to_summary"},
)
class CustomerRecord:
    def __init__(self, name: str, last_purchase: str, spend_ytd: float):
        self.name = name
        self.last_purchase = last_purchase
        self.spend_ytd = spend_ytd

    def to_summary(self, summary: str) -> "CustomerRecord":
        """Return a new CustomerRecord with the name replaced by the given summary."""
        return CustomerRecord(summary, self.last_purchase, self.spend_ytd)

record = CustomerRecord("Ada", "wireless headphones", 1240.50)
m = mellea.start_session()

transformed = m.transform(record, "Write a one-line CRM note for this customer.")
print(str(transformed))
# Output will vary — LLM responses depend on model and temperature.
```

The LLM calls `to_summary(summary=...)` with the generated text, and the return
value of that method is the result.

## Step 5: Mify an object ad hoc

You can also mify an existing object instance without decorating its class — useful
when you don't own the class definition:

```python
# Requires: mellea
# Returns: str
import mellea
from mellea.stdlib.components.mify import mify

class ThirdPartyRecord:
    def __init__(self, name: str, value: float):
        self.name = name
        self.value = value

record = ThirdPartyRecord("Acme Corp", 58000.0)
mify(record)  # adds the MifiedProtocol to this instance only

m = mellea.start_session()
result = m.query(record, "Is this a large or small account?")
print(str(result))
# Output will vary — LLM responses depend on model and temperature.
```

## What you built

A set of patterns for making legacy Python objects LLM-queryable without
modifying their class definitions:

| Pattern | Use when |
| --- | --- |
| `@mify` (default) | All fields can be exposed |
| `stringify_func` | Custom text representation needed |
| `fields_include` + `template` | Only a subset of fields should be visible |
| `funcs_include` | Specific methods should be callable by the LLM |
| `mify(obj)` | You don't own the class |

**See also:** [MObjects and mify](../concepts/mobjects-and-mify) |
[Working with Data](../how-to/working-with-data) |
[Tutorial 03: Using Generative Stubs](./03-using-generative-stubs)
