# Adapter Function Examples

Examples demonstrating Mellea's adapter functions: specialized LoRA/aLoRA-backed
capabilities (RAG quality checks, safety checks, requirement validation) accessed
through `LocalHFBackend`, `OpenAIBackend` with Granite Switch, or Ollama. For
setup, backend support, and a conceptual overview, see
[Adapter functions](../../docs/advanced/intrinsics.md) and
[Safety guardrails](../../docs/how-to/safety-guardrails.md). To run the same
adapter functions on a Granite Switch model, see [`granite-switch/`](../granite-switch/README.md).
To train your own, see [`aLora/`](../aLora/README.md).

## Examples

| File | Description |
| :--- | :--- |
| [`answerability.py`](./answerability.py) | `rag.check_answerability()`: whether retrieved documents can answer a question. |
| [`citations.py`](./citations.py) | `rag.find_citations()`: document sentences that support each sentence of a response. |
| [`context_attribution.py`](./context_attribution.py) | `core.find_context_attributions()`: context sentences that most influenced each response sentence. |
| [`hallucination_detection.py`](./hallucination_detection.py) | `rag.flag_hallucinated_content()`: response sentences not supported by the documents. |
| [`query_clarification.py`](./query_clarification.py) | `rag.clarify_query()`: a clarification request for an ambiguous query, or "CLEAR". |
| [`query_rewrite.py`](./query_rewrite.py) | `rag.rewrite_question()`: rewriting a follow-up question into a standalone retrieval query. |
| [`requirement_check.py`](./requirement_check.py) | `core.requirement_check()`: scoring whether a response satisfies a requirement (the check behind `ALoraRequirement`). |
| [`uncertainty.py`](./uncertainty.py) | `core.check_certainty()`: model certainty about its own answer, on `LocalHFBackend`. |
| [`uncertainty_ollama.py`](./uncertainty_ollama.py) | `core.check_certainty()` on `OllamaModelBackend` using an Ollama model that bundles the uncertainty aLoRA. |
| [`factuality_detection.py`](./factuality_detection.py) | `guardian.factuality_detection()`: "yes" if a response is factually incorrect given a document, "no" otherwise. |
| [`factuality_correction.py`](./factuality_correction.py) | `guardian.factuality_correction()`: rewriting a response so it agrees with a document. |
| [`guardian_core.py`](./guardian_core.py) | `guardian.guardian_check()`: harm, social bias, groundedness, custom criteria, function-call hallucination, and answer relevance. |
| [`policy_guardrails.py`](./policy_guardrails.py) | `guardian.policy_guardrails()`: whether a scenario complies with a policy text. |
| [`intrinsics.py`](./intrinsics.py) | Direct use of an adapter function via a composed `Adapter`, `Intrinsic`, and `mfuncs.act()`, underneath the helper functions above. |
