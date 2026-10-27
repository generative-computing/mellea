# Adapter Functions Examples

This directory contains examples for using Mellea's adapter functions - specialized
model capabilities accessed through adapters. For setup, backend support, and a
conceptual overview, see [Adapter functions](../../docs/advanced/intrinsics.md) and
[Safety guardrails](../../docs/how-to/safety-guardrails.md). Not all adapter
functions are embedded in every Granite Switch model; for Granite Switch examples,
see [`granite-switch/`](../granite-switch/README.md). To train your own adapter, see
[`aLora/`](../aLora/README.md).

## Examples

| File | Description |
| :--- | :--- |
| [`answerability.py`](./answerability.py) | Demonstrates checking if documents can answer a question. |
| [`citations.py`](./citations.py) | Demonstrates extracting and validating citations from responses. |
| [`context_attribution.py`](./context_attribution.py) | Demonstrates identifying which context sentences influenced the response. |
| [`hallucination_detection.py`](./hallucination_detection.py) | Demonstrates detecting potentially hallucinated content. |
| [`query_clarification.py`](./query_clarification.py) | Demonstrates generating clarification requests when needed. |
| [`query_rewrite.py`](./query_rewrite.py) | Demonstrates rewriting queries for better retrieval. |
| [`requirement_check.py`](./requirement_check.py) | Demonstrates validating requirements against model outputs. |
| [`uncertainty.py`](./uncertainty.py) | Demonstrates assessing model certainty about its response. |
| [`uncertainty_ollama.py`](./uncertainty_ollama.py) | `core.check_certainty()` on `OllamaModelBackend` using an Ollama model that bundles the uncertainty aLoRA. |
| [`factuality_detection.py`](./factuality_detection.py) | Demonstrates detecting factually incorrect statements. |
| [`factuality_correction.py`](./factuality_correction.py) | Demonstrates automatically correcting factual errors. |
| [`guardian_core.py`](./guardian_core.py) | Demonstrates comprehensive safety checks (harm, bias, groundedness). |
| [`policy_guardrails.py`](./policy_guardrails.py) | Demonstrates checking compliance with policies. |
| [`intrinsics.py`](./intrinsics.py) | Direct use of an adapter function via a composed `Adapter`, `Intrinsic`, and `mfuncs.act()`, underneath the helper functions above. |

## Architecture

![Granite Libraries Software Stack Architecture in Mellea](../../static/images/granite-libraries-mellea-architecture.png)
