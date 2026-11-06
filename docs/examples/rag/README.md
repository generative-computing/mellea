# RAG Examples

Examples demonstrating Retrieval-Augmented Generation (RAG) patterns in Mellea.
For a step-by-step guide to building a full pipeline, see the
[Build a RAG Pipeline](../../docs/how-to/build-a-rag-pipeline.md) how-to.

## Examples

| File | Description |
| :--- | :--- |
| [`simple_rag_with_filter.py`](./simple_rag_with_filter.py) | Two-stage pipeline: FAISS semantic search followed by a `@generative` relevance filter, then grounded answer generation with `grounding_context`. Requires `faiss-cpu` and `sentence-transformers`. |
| [`mellea_pdf.py`](./mellea_pdf.py) | Document-grounded question answering using [W3C's WCAG 1.0 checklist PDF](https://www.w3.org/TR/1999/WAI-WEBCONTENT-19990505/full-checklist.pdf). Requires `docling`. |
