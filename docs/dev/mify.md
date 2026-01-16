# mify

In classical programming, object-orientation provides a way to couple data and functionality.
Classes have fields and methods. Fields store data and methods operate over that data.

The mellea library allows you to interface with objects in the same way, but with the added benefit that an LLM can perform operations for you.

```python
import mellea

m = mellea.start_session()


class Circle:
    """A circle is defined by its center and a radius."""
    center_x: float
    center_y: float
    radius: float


c = Circle(1, 0, 1)

mify(c)

# .query is used to compute things.
circumference: float = m.query(c, "compute the circumference of the circle",
                               format=float)

# .transform is used to create a new class of the same type but mutated.
flipped_circle = m.transform(c, "Mirror the circle across the y axis.")
```

Let's consider a slightly more complicated example.

```python
class Customer:
    customer_id: int
    name: str
    age: int
    email_addr: str
    employer: str
    meeting_notes: List[str]

    def __init__(customer_id: int):
        ...

    def send_email(subject: str, body: str):
        ...

    def get_meeting_notes() -> List[str]:
        ...
```

...

```python
ctx = mellea.SingleShotContext(backend=WatsonX("ibm/granite4"))

customer = Customer(customer_id=42)
mify(c)

meetings_summary = m.query(c, "Summarize the last three interactions with this customer.")

email_body = ctx.instruct("Based upon the summary of notes from recent meetings, write an email body encouraging the customer to purchase three cases of self-sealing stembolts", grounding_context={"meetings_summary": meetings_summary})

email_subject = ctx.instruct("Write a subject for this sales email.", grounding_context={"email_body": email_body})

customer.execute("send an email.", email_body, email_subject)
```

For more examples and information, see
- [Mify Examples](../examples/mify.py)
- [Mify Implementation](../../mellea/stdlib/mify.py)
