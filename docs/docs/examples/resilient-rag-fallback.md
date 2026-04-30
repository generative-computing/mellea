---
canonical: "https://docs.mellea.ai/examples/resilient-rag-fallback"
title: "Resilient RAG with Fallback Filtering"
description: "Build a retrieval-augmented generation pipeline that uses FAISS for vector search and a @generative relevance filter to remove noise before generation."
# diataxis: reference
---

This example builds a complete RAG pipeline in three stages: embed and index a
document corpus, retrieve candidates by semantic similarity, then use a
[`@generative`](../reference/glossary#generative) boolean function to discard irrelevant candidates before passing
the survivors to a grounded `m.instruct()` call.

**Source file:** `docs/examples/rag/simple_rag_with_filter.py`

## Concepts covered

- Building a FAISS flat inner-product index from sentence-transformer embeddings
- Using `@generative` returning `bool` as a per-document relevance gate
- Passing filtered documents as [`grounding_context`](../reference/glossary#grounding_context) to `m.instruct()`
- Running the example with `uv run` via an inline PEP 723 dependency block

## Prerequisites

- [Quick Start](../getting-started/quickstart) complete
- `faiss-cpu` and `sentence-transformers` installed, **or** run via `uv run`
  which installs them automatically from the inline script block
- Ollama running locally with `granite4:micro` pulled (or a Mistral model — see
  the session setup section below)

Install dependencies manually if you are not using `uv run`:

```bash
pip install faiss-cpu sentence-transformers
```

## Pipeline architecture

```text
Query
  |
  v
Embedding model  (sentence-transformers all-MiniLM-L6-v2)
  |
  v
FAISS vector search  (top-k candidates)
  |
  v
@generative relevance filter  (per-document boolean check)
  |
  v
m.instruct() with grounding_context  (answer generation)
  |
  v
Final answer
```

## The full example

### Inline script dependencies

```python
# Requires: faiss-cpu, sentence-transformers, mellea
# Returns: N/A
# pytest: skip_always
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "faiss-cpu",
#     "sentence_transformers",
#     "mellea"
# ]
# ///
```

The `/// script` block follows [PEP 723](https://peps.python.org/pep-0723/).
When you run the file with `uv run simple_rag_with_filter.py`, `uv` reads this
block and installs the listed packages into a temporary environment before
execution. No manual `pip install` is needed.

### Imports and document corpus

```python
# Requires: faiss-cpu, sentence-transformers, mellea
# Returns: list[str]
from faiss import IndexFlatIP
from sentence_transformers import SentenceTransformer

from mellea import generative, start_session
from mellea.backends import model_ids

docs = [
    "The capital of France is Paris. Paris is known for its Eiffel Tower.",
    "The Amazon River is the largest river by discharge volume of water in the world.",
    "Mount Everest is the Earth's highest mountain above sea level, located in the Himalayas.",
    "The Louvre Museum in Paris houses the Mona Lisa.",
    "Artificial intelligence (AI) is intelligence demonstrated by machines.",
    "Machine learning is a subset of AI that enables systems to learn from data.",
    "Natural Language Processing (NLP) is a field of AI that focuses on enabling computers to understand, process, and generate human language.",
    "The Great Wall of China is a series of fortifications made of stone, brick, tamped earth, wood, and other materials, generally built along an east-to-west line across the historical northern borders of China.",
    "The solar system consists of the Sun and everything bound to it by gravity, including the eight planets, dwarf planets, and countless small Solar System bodies.",
    "Mars is the fourth planet from the Sun and the second-smallest planet in the Solar System, after Mercury.",
    "The human heart has four chambers: two atria and two ventricles.",
    "Photosynthesis is the process used by plants, algae, and cyanobacteria to convert light energy into chemical energy.",
    "The internet is a global system of interconnected computer networks that uses the Internet protocol suite (TCP/IP) to communicate between networks and devices.",
    "Python is a high-level, general-purpose programming language.",
    "The Pacific Ocean is the largest and deepest of Earth's five oceanic divisions.",
]
```

The corpus is a flat list of strings. In a real system these would come from a
database, file system, or document store. `IndexFlatIP` is a FAISS index that
scores by inner product — equivalent to cosine similarity when the embeddings
are L2-normalised, as `sentence-transformers` produces by default.

### Index creation and querying

```python
# Requires: faiss-cpu, sentence-transformers
# Returns: IndexFlatIP
def create_index(model, ds: list[str]) -> IndexFlatIP:
    print("running encoding... ")
    embeddings = model.encode(ds)
    print("running embeddings... ")
    dimension = embeddings.shape[1]
    index = IndexFlatIP(dimension)
    index.add(embeddings)  # type:ignore
    print("done indexing.")
    return index


def query_index(model, idx: IndexFlatIP, query: str, ds: list[str], k: int = 5) -> list:
    query_embedding = model.encode([query])
    _distances, indices = idx.search(query_embedding, k=k)
    return [ds[i] for i in indices[0]]
```

`create_index` encodes all documents once and stores the result. `query_index`
encodes the query at inference time and returns the top-`k` documents by
similarity. The default `k=5` gives the filter stage enough candidates without
overwhelming the context window.

### The relevance filter

```python
# Requires: mellea
# Returns: bool
@generative
def is_answer_relevant_to_question(answer: str, question: str) -> bool:
    """For the given question, determine whether the answer is relevant or not."""
```

A `@generative` function returning `bool` acts as a classifier. The docstring
frames the task: given a candidate document (`answer`) and the original query
(`question`), decide whether the document is actually useful.

Vector similarity finds documents that are *topically related*, but it can
return documents that mention the same keywords without actually answering the
question. This LLM filter catches those false positives.

### Main: retrieval, filtering, and generation

```python
if __name__ == "__main__":
    query = "How are AI and NLP related?"

    # Create a simple embedding index
    print("loading Embedding model and index data...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    index = create_index(embedding_model, docs)

    # Query the index
    print("Query Embedding model...")
    results = query_index(embedding_model, index, query, docs)
    results_str = "\n".join([f"=> {r}" for r in results])
    print(f"results:\n {results_str}\n ====")
    del embedding_model  # help GC

    # Create Mellea session with Mistral. Also work with other models.
    m = start_session(model_id=model_ids.MISTRALAI_MISTRAL_0_3_7B)

    # Check for each document from retrieval if it is actually relevant
    print("running filter.. ")
    relevant_answers = []
    for doc in results:
        is_it = is_answer_relevant_to_question(m, answer=doc, question=query)
        if is_it:
            relevant_answers.append(doc)
        else:
            print(f"skipping: {doc}")

    # Run final answer generation from here
    print("running generation...")
    answer = m.instruct(
        "Provided the documents in the context, answer the question: `{{query}}`",
        user_variables={"query": query},
        grounding_context={f"doc{i}": doc for i, doc in enumerate(relevant_answers)},
    )

    # Print results answer
    print(f"== answer == \n{answer.value}\n ====")
```

Several implementation choices are worth noting:

**`del embedding_model`** frees the sentence-transformer weights before loading
the LLM backend. On a machine with limited VRAM or RAM this prevents
out-of-memory errors when both models would otherwise be resident simultaneously.

**`model_id=model_ids.MISTRALAI_MISTRAL_0_3_7B`** selects a specific backend
model. You can substitute any model constant from `model_ids` or pass a string
identifier directly. The example comment confirms other models work too.

**`grounding_context`** passes the surviving documents as named context
entries. The template variable `{{query}}` is supplied separately via
`user_variables`. Keeping query and context separate lets Mellea render the
prompt correctly and trace each component independently.

**`answer.value`** retrieves the raw string from the
[`ModelOutputThunk`](../reference/glossary#modeloutputthunk) returned by
`m.instruct()`.

### Full file

```python
# Requires: faiss-cpu, sentence-transformers, mellea
# Returns: N/A
# pytest: skip_always
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "faiss-cpu",
#     "sentence_transformers",
#     "mellea"
# ]
# ///
"""
Simple RAG (Retrieval-Augmented Generation) example with relevance filtering.

This script demonstrates how to:
1. Create a FAISS vector index from documents
2. Retrieve relevant documents using semantic search
3. Filter retrieved documents for relevance using Mellea
4. Generate a final answer based on the filtered documents

Use `uv run simple_rag_with_filter.py` to run the script.
"""

from faiss import IndexFlatIP
from sentence_transformers import SentenceTransformer

from mellea import generative, start_session
from mellea.backends import model_ids

docs = [
    "The capital of France is Paris. Paris is known for its Eiffel Tower.",
    "The Amazon River is the largest river by discharge volume of water in the world.",
    "Mount Everest is the Earth's highest mountain above sea level, located in the Himalayas.",
    "The Louvre Museum in Paris houses the Mona Lisa.",
    "Artificial intelligence (AI) is intelligence demonstrated by machines.",
    "Machine learning is a subset of AI that enables systems to learn from data.",
    "Natural Language Processing (NLP) is a field of AI that focuses on enabling computers to understand, process, and generate human language.",
    "The Great Wall of China is a series of fortifications made of stone, brick, tamped earth, wood, and other materials, generally built along an east-to-west line across the historical northern borders of China.",
    "The solar system consists of the Sun and everything bound to it by gravity, including the eight planets, dwarf planets, and countless small Solar System bodies.",
    "Mars is the fourth planet from the Sun and the second-smallest planet in the Solar System, after Mercury.",
    "The human heart has four chambers: two atria and two ventricles.",
    "Photosynthesis is the process used by plants, algae, and cyanobacteria to convert light energy into chemical energy.",
    "The internet is a global system of interconnected computer networks that uses the Internet protocol suite (TCP/IP) to communicate between networks and devices.",
    "Python is a high-level, general-purpose programming language.",
    "The Pacific Ocean is the largest and deepest of Earth's five oceanic divisions.",
]


def create_index(model, ds: list[str]) -> IndexFlatIP:
    print("running encoding... ")
    embeddings = model.encode(ds)
    print("running embeddings... ")
    dimension = embeddings.shape[1]
    index = IndexFlatIP(dimension)
    index.add(embeddings)  # type:ignore
    print("done indexing.")
    return index


def query_index(model, idx: IndexFlatIP, query: str, ds: list[str], k: int = 5) -> list:
    query_embedding = model.encode([query])
    _distances, indices = idx.search(query_embedding, k=k)
    return [ds[i] for i in indices[0]]


@generative
def is_answer_relevant_to_question(answer: str, question: str) -> bool:
    """For the given question, determine whether the answer is relevant or not."""


if __name__ == "__main__":
    query = "How are AI and NLP related?"

    # Create a simple embedding index
    print("loading Embedding model and index data...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    index = create_index(embedding_model, docs)

    # Query the index
    print("Query Embedding model...")
    results = query_index(embedding_model, index, query, docs)
    results_str = "\n".join([f"=> {r}" for r in results])
    print(f"results:\n {results_str}\n ====")
    del embedding_model  # help GC

    # Create Mellea session with Mistral. Also work with other models.
    m = start_session(model_id=model_ids.MISTRALAI_MISTRAL_0_3_7B)

    # Check for each document from retrieval if it is actually relevant
    print("running filter.. ")
    relevant_answers = []
    for doc in results:
        is_it = is_answer_relevant_to_question(m, answer=doc, question=query)
        if is_it:
            relevant_answers.append(doc)
        else:
            print(f"skipping: {doc}")

    # Run final answer generation from here
    print("running generation...")
    answer = m.instruct(
        "Provided the documents in the context, answer the question: `{{query}}`",
        user_variables={"query": query},
        grounding_context={f"doc{i}": doc for i, doc in enumerate(relevant_answers)},
    )

    # Print results answer
    print(f"== answer == \n{answer.value}\n ====")
```

## Key observations

**Two-stage retrieval reduces hallucination.** Vector search alone can surface
documents that share vocabulary with the query but do not answer it. The LLM
filter adds a semantic gate that vector distance cannot provide.

**`@generative` returning `bool` is a classifier.** You can use this pattern
wherever you need a binary decision: spam detection, content moderation, input
validation, feature flags driven by natural language.

**`grounding_context` is the RAG anchor.** Without it, `m.instruct()` would
generate from the model's parametric knowledge. Passing documents through
`grounding_context` grounds the answer in retrieved evidence.

## What to try next

- Replace the in-memory list with a database-backed corpus and see
  `docs/examples/rag/mellea_pdf.py` for a PDF-based variant.
- Tune `k` in `query_index` and observe how the filter step affects final
  answer quality.
- Add `requirements` to the final `m.instruct()` call to enforce length,
  citation, or tone constraints — see the
  [requirements system concept](../concepts/requirements-system).

---

**See also:** [Build a RAG Pipeline](../how-to/build-a-rag-pipeline) — step-by-step how-to guide | [Examples Index](./index)
