---
canonical: "https://docs.mellea.ai/guide/working-with-data"
title: "Working with Data"
description: "Ground instructions with documents, build RAG pipelines, and use MObjects and RichDocument."
# diataxis: how-to
---

**Prerequisites:** [Quick Start](../getting-started/quickstart) complete, `pip install mellea`,
Ollama running locally. RAG examples require `faiss-cpu` and `sentence-transformers`.
`RichDocument` requires `pip install "mellea[docling]"` or `docling` installed separately.

## Grounding context

Attach reference documents to any `instruct()` call via `grounding_context`. The dict
maps string keys to document text injected as reference material into the prompt:

```python
from mellea import start_session

doc0 = "Artificial intelligence (AI) is intelligence demonstrated by machines."
doc1 = "Natural Language Processing (NLP) is a field of AI focused on human language."

m = start_session()
answer = m.instruct(
    "Given the documents in the context, answer: {{query}}",
    user_variables={"query": "How are AI and NLP related?"},
    grounding_context={"doc0": doc0, "doc1": doc1},
)
print(str(answer))
# Output will vary — LLM responses depend on model and temperature.
```

## RAG with relevance filtering

Combine vector retrieval with `@generative` relevance filtering for full RAG:

```python
from faiss import IndexFlatIP
from sentence_transformers import SentenceTransformer
from mellea import generative, start_session
from mellea.backends import model_ids

docs = [
    "Artificial intelligence (AI) is intelligence demonstrated by machines.",
    "Machine learning is a subset of AI that enables systems to learn from data.",
    "Natural Language Processing (NLP) is a field of AI focused on human language.",
]

# Build a FAISS embedding index
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(docs)
index = IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

# Retrieve top-k candidates
query = "How are AI and NLP related?"
query_emb = embedding_model.encode([query])
_, indices = index.search(query_emb, k=5)
candidates = [docs[i] for i in indices[0]]

# Filter for relevance using a generative function
@generative
def is_relevant(answer: str, question: str) -> bool:
    """Determine whether the answer is relevant to the question."""

m = start_session(model_id=model_ids.IBM_GRANITE_3_3_8B)
relevant_docs = [doc for doc in candidates if is_relevant(m, answer=doc, question=query)]

# Generate final answer from filtered documents
result = m.instruct(
    "Given the documents in the context, answer: {{query}}",
    user_variables={"query": query},
    grounding_context={f"doc{i}": doc for i, doc in enumerate(relevant_docs)},
)
print(str(result))
# Output will vary — LLM responses depend on model and temperature.
```

The `@generative` filter returns a typed `bool`, giving you deterministic branching
over LLM relevance judgments.

> **Full example:** [`docs/examples/rag/simple_rag_with_filter.py`](https://github.com/generative-computing/mellea/blob/main/docs/examples/rag/simple_rag_with_filter.py)

## MObjects — making data LLM-aware

The `@mify` decorator wraps any Python class so Mellea sessions can query and
transform its instances. This is the **MObject** pattern: store data alongside the
operations that apply to it, and expose both to the LLM in a controlled way.

```python
from mellea import start_session
from mellea.stdlib.components.mify import mify

@mify(fields_include={"table"}, template="{{ table }}")
class SalesDatabase:
    table: str = (
        "| Store     | Sales |\n"
        "| --------- | ----- |\n"
        "| Northeast | $250  |\n"
        "| Southeast | $80   |\n"
        "| Midwest   | $420  |"
    )

m = start_session()
db = SalesDatabase()
answer = m.query(db, "Which region had the highest sales?")
print(str(answer))
# Output will vary — LLM responses depend on model and temperature.
```

`fields_include` controls which fields are visible to the LLM. `template` controls
how the object is formatted in the prompt.

> **Full example:** [`docs/examples/tutorial/table_mobject.py`](https://github.com/generative-computing/mellea/blob/main/docs/examples/tutorial/table_mobject.py)

### `query()` and `transform()`

`m.query()` asks a question about an MObject. `m.transform()` asks the model to
produce a modified version:

```python
from mellea import start_session
from mellea.stdlib.components.mify import mify

@mify(fields_include={"table"}, template="{{ table }}")
class SalesDatabase:
    table: str = (
        "| Store     | Sales |\n"
        "| --------- | ----- |\n"
        "| Northeast | $250  |\n"
        "| Southeast | $80   |\n"
        "| Midwest   | $420  |"
    )

    def transpose(self) -> str:
        """Transpose the table rows and columns."""
        ...  # your implementation

m = start_session()
db = SalesDatabase()

# Ask a question
answer = m.query(db, "What were Northeast branch sales?")
print(str(answer))

# Request a transformation
transposed = m.transform(db, "Transpose the table.")
print(str(transposed))
# Output will vary — LLM responses depend on model and temperature.
```

When a mified class has methods with docstrings, they are registered as tools during
`transform()`. The LLM can call `transpose()` directly rather than generating the
transformation from scratch.

### Mifying an existing object ad-hoc

You can mify any existing object at call time without decorating the class:

```python
from mellea import start_session
from mellea.stdlib.components.mify import mify

class Store:
    def __init__(self, purchases: list[str]) -> None:
        self.purchases = purchases

m = start_session()
store = Store(["Beans", "Soil", "Watering Can"])
mify(store)
answer = m.query(store, "What was the most recent purchase?")
print(str(answer))
# Output will vary — LLM responses depend on model and temperature.
```

### Custom stringify

By default, mified objects use `__str__`. Override with `stringify_func`:

```python
from mellea.stdlib.components.mify import mify

@mify(stringify_func=lambda x: f"Location: {x.location}, Manager: {x.manager}")
class Branch:
    def __init__(self, location: str, manager: str) -> None:
        self.location = location
        self.manager = manager
```

### Controlling exposed methods

Use `funcs_include` or `funcs_exclude` to control which methods the LLM can call:

```python
from mellea import start_session
from mellea.stdlib.components.mify import mify

@mify(funcs_include={"from_markdown"})
class DocumentLoader:
    def __init__(self) -> None:
        self.content = ""

    @classmethod
    def from_markdown(cls, text: str) -> "DocumentLoader":
        """Load document content from a Markdown string."""
        doc = DocumentLoader()
        doc.content = text
        return doc

    def internal_helper(self) -> str:
        """Not exposed to the LLM."""
        return "internal"

m = start_session()
result = m.transform(DocumentLoader(), "Write a haiku about mountains.")
print(str(result))
# Output will vary — LLM responses depend on model and temperature.
```

## RichDocument — working with PDFs and structured documents

> **Backend note:** `RichDocument` requires the `docling` library:
> `pip install docling`. First-time use downloads parser models.

`RichDocument` loads and parses PDFs and other documents into a Mellea-ready
structure, including extractable tables:

```python
from mellea import start_session
from mellea.stdlib.components.docs.richdocument import RichDocument, Table

rd = RichDocument.from_document_file("path/to/document.pdf")

# Extract the first table
tables = rd.get_tables()
if tables:
    table: Table = tables[0]
    print(table.to_markdown())

    # Transform it with the LLM
    m = start_session()
    updated = m.transform(table, "Add a 'Total' row summing all sales values.")
    print(str(updated))
    # Output will vary — LLM responses depend on model and temperature.
```

`Table` is itself an MObject — its methods (e.g., `transpose()`) are registered as
tools during `transform()` calls automatically.

> **Full example:** [`docs/examples/tutorial/document_mobject.py`](https://github.com/generative-computing/mellea/blob/main/docs/examples/tutorial/document_mobject.py)

---

**See also:** [act() and aact()](../how-to/act-and-aact) | [MObjects and mify](../concepts/mobjects-and-mify)
