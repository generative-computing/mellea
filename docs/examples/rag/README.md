# RAG (Retrieval-Augmented Generation) Examples

This directory contains examples of implementing RAG systems with Mellea.

## Files

### [simple_rag_with_filter.py](./simple_rag_with_filter.py)
A complete RAG pipeline with relevance filtering using generative stubs.

**Key Features:**
- Vector embedding with FAISS index
- Semantic search for document retrieval
- LLM-based relevance filtering
- Answer generation with grounded context
- Using `@generative` for filtering logic

### [mellea_pdf.py](./mellea_pdf.py)
Document-grounded question answering using W3C's
[WCAG 1.0 checklist PDF](https://www.w3.org/TR/1999/WAI-WEBCONTENT-19990505/full-checklist.pdf).
The question asks what the three checkpoint priority levels mean; the answer is
in the document's **Priorities** section on page 2. In brief, developers must
satisfy Priority 1 checkpoints, should satisfy Priority 2 checkpoints, and may
address Priority 3 checkpoints. Generated wording can vary.

This dated, seven-page publication is historical sample material, not current
accessibility guidance. It replaces a USDA PDF that returned HTTP 403 for users
outside the US. The PDF remains hosted by W3C under its
[document license](https://www.w3.org/Consortium/Legal/copyright-documents-19990405);
no copy is bundled with this example.

**Prerequisites:** Install the Docling extra, have internet access to download
the PDF and Docling's conversion models, and run Ollama with `granite4.2:3b`
(the current `start_session()` default). FAISS and sentence-transformers are
not needed for this example. OCR is disabled because the PDF contains text;
Docling may still download layout and table models on the first conversion.

From the repository root:

```bash
uv sync --frozen --extra docling
ollama serve  # run in a separate terminal if Ollama is not already running
ollama pull granite4.2:3b
uv run python docs/examples/rag/mellea_pdf.py
```

From another working directory, use absolute paths to the checkout and script:

```bash
uv run --project /path/to/mellea python /path/to/mellea/docs/examples/rag/mellea_pdf.py
```

The document URL is independent of the working directory. PDF conversion happens
before the session is created, so document-loading failures can be distinguished
from model-inference failures. The `e2e, ollama, slow` markers opt the example
into explicit slow-test runs. Its assertions check only that the result is a
non-empty string, not the factual content of the answer. The `slow` marker
accounts for cold conversion-model downloads and inference; ordinary test runs
exclude it. `CICD=1` does not skip this structural test, but current PR CI does
not collect Python examples.

To run it through pytest, including the slow marker:

```bash
uv run pytest docs/examples/rag/mellea_pdf.py -m "e2e and ollama" --no-cov -v -s
```

## Concepts Demonstrated

- **Vector Search**: Using embeddings for semantic retrieval
- **Relevance Filtering**: Using LLMs to filter retrieved documents
- **Grounded Generation**: Generating answers based on retrieved context
- **Multi-Stage Pipeline**: Retrieval → Filtering → Generation
- **Generative Stubs**: Using `@generative` for RAG components

## Pipeline Architecture

```
Query
  ↓
Embedding Model (sentence-transformers)
  ↓
Vector Search (FAISS)
  ↓
Retrieved Documents
  ↓
Relevance Filter (LLM)
  ↓
Relevant Documents
  ↓
Answer Generation (LLM with grounding_context)
  ↓
Final Answer
```

## Basic Usage

```python
from mellea import generative, start_session
from sentence_transformers import SentenceTransformer
from faiss import IndexFlatIP

# 1. Create embeddings and index
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(documents)
index = IndexFlatIP(dimension)
index.add(embeddings)

# 2. Retrieve documents
query_embedding = embedding_model.encode([query])
distances, indices = index.search(query_embedding, k=5)
retrieved_docs = [documents[i] for i in indices[0]]

# 3. Filter for relevance
@generative
def is_answer_relevant_to_question(answer: str, question: str) -> bool:
    """Determine whether the answer is relevant to the question."""

m = start_session()
relevant_docs = [
    doc for doc in retrieved_docs 
    if is_answer_relevant_to_question(m, answer=doc, question=query)
]

# 4. Generate answer with grounding
answer = m.instruct(
    "Answer the question: {{query}}",
    user_variables={"query": query},
    grounding_context={f"doc{i}": doc for i, doc in enumerate(relevant_docs)}
)
```

## Dependencies

```bash
# Install RAG dependencies
uv pip install faiss-cpu sentence-transformers
```

## Key Techniques

- **Semantic Search**: Finding relevant documents by meaning, not keywords
- **Two-Stage Retrieval**: Broad retrieval followed by precise filtering
- **Grounding Context**: Providing retrieved documents to the LLM
- **Relevance Validation**: Using LLMs to validate retrieval quality

## Related Documentation

- See `intrinsics/` for RAG evaluation adapter functions
- See `mini_researcher/` for a complete RAG application
- See `mellea/stdlib/components/intrinsic/rag.py` for RAG helpers
