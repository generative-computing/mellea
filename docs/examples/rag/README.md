# RAG (Retrieval-Augmented Generation) Examples

This directory contains examples of implementing RAG systems with Mellea.

## Files

### simple_rag_with_filter.py
A complete RAG pipeline with relevance filtering using generative slots.

**Key Features:**
- Vector embedding with FAISS index
- Semantic search for document retrieval
- LLM-based relevance filtering
- Answer generation with grounded context
- Using `@generative` for filtering logic

### mellea_pdf.py
RAG example specifically for PDF documents.

## Concepts Demonstrated

- **Vector Search**: Using embeddings for semantic retrieval
- **Relevance Filtering**: Using LLMs to filter retrieved documents
- **Grounded Generation**: Generating answers based on retrieved context
- **Multi-Stage Pipeline**: Retrieval → Filtering → Generation
- **Generative Slots**: Using `@generative` for RAG components

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

- See `intrinsics/` for RAG evaluation intrinsics
- See `mini_researcher/` for a complete RAG application
- See `mellea/stdlib/components/intrinsic/rag.py` for RAG helpers