---
title: "Build a RAG Pipeline"
description: "Combine vector retrieval with Mellea's generative filtering and grounded generation to build a reliable retrieval-augmented generation system."
# diataxis: how-to
---

**Prerequisites:** [Quick Start](../getting-started/quickstart) complete,
`pip install mellea faiss-cpu sentence-transformers`, Ollama running locally.

Retrieval-augmented generation (RAG) reduces hallucination by grounding the
model's answer in documents you supply. Mellea adds two things a plain RAG loop
lacks: an LLM-based relevance filter before generation, and optional
groundedness checking after.

---

## The pipeline

```text
Query
  |
  v
Embedding model  →  vector search  →  top-k candidates
                                            |
                                            v
                              @generative relevance filter
                                            |
                                            v
                            m.instruct() with grounding_context
                                            |
                                            v
                                       Final answer
                              (optional: GuardianCheck groundedness)
```

---

## Step 1: Index your documents

Use any embedding model and vector store. This example uses
`sentence-transformers` and a FAISS flat inner-product index:

```python
from faiss import IndexFlatIP
from sentence_transformers import SentenceTransformer

def build_index(docs: list[str], model: SentenceTransformer) -> IndexFlatIP:
    embeddings = model.encode(docs)
    index = IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)  # type: ignore
    return index

def search(
    query: str,
    docs: list[str],
    index: IndexFlatIP,
    model: SentenceTransformer,
    k: int = 5,
) -> list[str]:
    query_vec = model.encode([query])
    _, indices = index.search(query_vec, k)
    return [docs[i] for i in indices[0]]
```

`IndexFlatIP` scores by inner product, which is equivalent to cosine similarity
for L2-normalised embeddings — the default output of `sentence-transformers`.

**Choosing `k`:** start with 5. Too small risks missing the relevant document;
too large floods the filter step and the context window. Tune after measuring
filter acceptance rates.

---

## Step 2: Filter candidates with `@generative`

Vector similarity finds *topically related* documents but cannot determine
whether a document actually answers the question. Add an LLM filter:

```python
from mellea import generative

@generative
def is_relevant(document: str, question: str) -> bool:
    """Determine whether the document contains information that would help answer the question."""
```

Apply it after retrieval:

```python
import mellea

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
index = build_index(docs, embedding_model)
candidates = search(query, docs, index, embedding_model)
del embedding_model  # free memory before loading the LLM

m = mellea.start_session()

relevant = [
    doc for doc in candidates
    if is_relevant(m, document=doc, question=query)
]
```

`del embedding_model` before starting the Mellea session avoids having both
models resident simultaneously — important on memory-constrained machines.

If all candidates are filtered out, fall back gracefully rather than calling
`m.instruct()` with an empty context:

```python
if not relevant:
    print("No relevant documents found.")
else:
    # proceed to generation
    ...
```

---

## Step 3: Generate with `grounding_context`

Pass the surviving documents as named entries in `grounding_context`. Mellea
injects them into the prompt and tracks them as separate context components:

```python
answer = m.instruct(
    "Using the provided documents, answer the following question: {{question}}",
    user_variables={"question": query},
    grounding_context={f"doc{i}": doc for i, doc in enumerate(relevant)},
)
print(str(answer))
```

`grounding_context` is separate from `user_variables` so each component is
rendered and traced independently. Without it, `m.instruct()` generates from
the model's parametric knowledge — no grounding.

---

## Step 4: Add requirements to the answer (optional)

Use `requirements` to enforce answer format, length, or citation style:

```python
from mellea.stdlib.requirements import req, simple_validate

answer = m.instruct(
    "Using the provided documents, answer the following question: {{question}}",
    user_variables={"question": query},
    grounding_context={f"doc{i}": doc for i, doc in enumerate(relevant)},
    requirements=[
        req("The answer must be based only on the provided documents."),
        req(
            "The answer must be 100 words or fewer.",
            validation_fn=simple_validate(
                lambda x: (
                    len(x.split()) <= 100,
                    f"Answer is {len(x.split())} words; must be 100 or fewer.",
                )
            ),
        ),
    ],
)
```

---

## Step 5: Check groundedness (optional)

After generation, use `GuardianCheck` with `GuardianRisk.GROUNDEDNESS` to
verify the answer does not hallucinate beyond the retrieved documents:

```python
from mellea.stdlib.requirements.safety.guardian import GuardianCheck, GuardianRisk

groundedness_check = GuardianCheck(
    GuardianRisk.GROUNDEDNESS,
    backend_type="ollama",
    ollama_url="http://localhost:11434",
    context_text="\n\n".join(relevant),
)

results = m.validate([groundedness_check])
if results[0]._result:
    print("Grounded answer:", str(answer))
else:
    print("Answer may contain hallucinated content:", results[0]._reason)
```

Pass the same text to `context_text` that you used in `grounding_context` —
this ensures the groundedness model evaluates the answer against exactly what
the generator was given.

> **Note:** `GuardianCheck` requires `granite3-guardian:2b` pulled in Ollama.
> Run `ollama pull granite3-guardian:2b` before using it.

---

## Putting it together

```python
from faiss import IndexFlatIP
from sentence_transformers import SentenceTransformer

import mellea
from mellea import generative
from mellea.stdlib.requirements import req, simple_validate
from mellea.stdlib.requirements.safety.guardian import GuardianCheck, GuardianRisk


@generative
def is_relevant(document: str, question: str) -> bool:
    """Determine whether the document contains information that would help answer the question."""


def build_index(docs: list[str], model: SentenceTransformer) -> IndexFlatIP:
    embeddings = model.encode(docs)
    index = IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)  # type: ignore
    return index


def search(query: str, docs: list[str], index: IndexFlatIP,
           model: SentenceTransformer, k: int = 5) -> list[str]:
    _, indices = index.search(model.encode([query]), k)
    return [docs[i] for i in indices[0]]


def rag(docs: list[str], query: str) -> str | None:
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    index = build_index(docs, embedding_model)
    candidates = search(query, docs, index, embedding_model)
    del embedding_model

    m = mellea.start_session()

    relevant = [doc for doc in candidates if is_relevant(m, document=doc, question=query)]
    if not relevant:
        return None

    answer = m.instruct(
        "Using the provided documents, answer this question: {{question}}",
        user_variables={"question": query},
        grounding_context={f"doc{i}": doc for i, doc in enumerate(relevant)},
        requirements=[req("Answer only from the provided documents.")],
    )

    results = m.validate([GuardianCheck(
        GuardianRisk.GROUNDEDNESS,
        backend_type="ollama",
        ollama_url="http://localhost:11434",
        context_text="\n\n".join(relevant),
    )])
    if not results[0]._result:
        print("Warning: groundedness check failed:", results[0]._reason)

    return str(answer)
```

---

## What to tune

| Parameter | Effect | Starting point |
| --------- | ------ | -------------- |
| `k` in `search()` | Candidates passed to the filter | 5 |
| `is_relevant` docstring | How strictly the filter interprets relevance | Adjust phrasing to match your domain |
| `grounding_context` key names | Tracing and debugging in spans | Use descriptive names in production |
| `requirements` on `m.instruct()` | Answer length, citation, tone | Add after baseline quality is good |
| GuardianCheck `context_text` | What the groundedness model checks against | Match exactly what you pass to `grounding_context` |

---

## See also

- [Resilient RAG with Fallback Filtering](../examples/resilient-rag-fallback) — annotated walkthrough of the complete source file
- [Making Agents Reliable](../tutorials/04-making-agents-reliable) — Guardian checks in depth
- [The Requirements System](../concepts/requirements-system) — advanced validators for the generation step
- `docs/examples/rag/` — runnable source including a PDF variant (`simple_rag_with_filter.py`, `mellea_pdf.py`)
