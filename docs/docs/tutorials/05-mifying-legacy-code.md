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
`pip install mellea`, Ollama running locally with `granite4:micro` downloaded.

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

@mify(stringify_func=lambda r: (
    f"name: {r.name}\n"
    f"last_purchase: {r.last_purchase}\n"
    f"spend_ytd: {r.spend_ytd}"
))
class CustomerRecord:
    def __init__(self, name: str, last_purchase: str, spend_ytd: float):
        self.name = name
        self.last_purchase = last_purchase
        self.spend_ytd = spend_ytd

record = CustomerRecord("Ada", "wireless headphones", 1240.50)

m = mellea.start_session()
result = m.query(record, "What was this customer's last purchase?")
print(str(result))
```

**Sample output:**

```text
The last purchase of the customer named Ada is wireless headphones.
```

> **Note:** LLM output is non-deterministic, output may vary.

`@mify` adds the [`MObject`](../reference/glossary#mobject) protocol to every instance. The
`stringify_func` controls the text the LLM receives. It is required here because
without it the model sees Python's default `str()` repr — which for a custom class
contains no field values — and cannot answer questions about the object's data.

> **Full example:** [`docs/examples/mify/mify.py`](https://github.com/generative-computing/mellea/blob/main/docs/examples/mify/mify.py)

## Step 2: Control the text representation

Refine the `stringify_func` to produce well-labelled, human-readable text that
gives the model the clearest possible view of the object:

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
```

**Sample output:**

```text
Based on the information provided in your query, it's difficult to
definitively categorize Ada as a "high-value" customer without
additional context or comparison data. High-value customers are
typically defined by their purchase frequency and average order
value (AOV), not just by total spend.

From the given details:
- Customer: Ada
- Last Purchase: Wireless Headphones
- Year-to-date Spend: £1240.50

There is no information available about how often Ada has made
purchases or what her typical spending per transaction looks like.
Therefore, it's impossible to say whether this customer qualifies
as high-value without more data.

For a better understanding of Ada's value, consider gathering the
following:
1. Total number of transactions over the year.
2. Average order value (AOV).
3. Time between purchases.

Once these details are available, you can analyze them against your
company's specific criteria for what constitutes a high-value customer.
```

> **Note:** LLM output is non-deterministic, output may vary.

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
```

**Sample output:**

```text
Based on the information provided — Ada spent £1240.50 this year —
this customer would fall into a low-value tier. Low-value customers
typically have annual spend under £2000, medium between £2000 and
£10000, and high above £10000.
```

> **Note:** LLM output is non-deterministic, output may vary. The model only sees
> `name` and `spend_ytd` — `last_purchase` is excluded by `fields_include` and will
> never appear in the response.

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
print(str(transformed.name))
```

**Sample output:**

```text
Ordered wireless headphones for home use.
```

> **Note:** `m.transform()` returns whatever `to_summary()` returns — here a new
> `CustomerRecord` constructed with the generated text as `name`. Access
> `.name` to retrieve it. Your wording will vary.

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
mify(record, stringify_func=lambda: f"name: {record.name}\nvalue: {record.value}")

m = mellea.start_session()
result = m.query(record, "Is this a large or small account?")
print(str(result))
```

**Sample output:**

```text
The term "large" or "small" for an account can be subjective and
dependent on various factors, such as industry standards, company
size, or specific business context.

However, if we consider the given value of Acme Corp to be 58,000.0,
this could potentially fall into a category typically defined as small
businesses in some industries, particularly in the U.S., where "small"
is often less than $6 million according to SBA guidelines for many
sectors.

Please note that these classifications can vary greatly based on the
industry and location. For a precise classification, it would be best
to compare this value against relevant benchmarks or consult with a
professional in your specific field.
```

> **Note:** LLM output is non-deterministic, output may vary.
>
> When `mify` is called on an **instance**, `stringify_func` must be a
> zero-argument callable that closes over the instance (as above). This differs from
> the class-level `@mify(stringify_func=lambda r: ...)` form, where Python's method
> binding passes the instance as the first argument. Using `lambda r:` on an instance
> will raise a `TypeError` at query time.

## What you built

A set of patterns for making legacy Python objects LLM-queryable without
modifying their class definitions:

| Pattern | Use when |
| --- | --- |
| `@mify(stringify_func=...)` | Baseline form — controls the text the LLM receives |
| `stringify_func` | Custom text representation needed |
| `fields_include` + `template` | Only a subset of fields should be visible |
| `funcs_include` | Specific methods should be callable by the LLM |
| `mify(obj)` | You don't own the class |

**See also:** [MObjects and mify](../concepts/mobjects-and-mify) |
[Working with Data](../how-to/working-with-data) |
[Tutorial 03: Using Generative Stubs](./03-using-generative-stubs)
