# Granite Switch Examples

Examples demonstrating Mellea adapter functions on Granite Switch models, whose
LoRA/aLoRA adapters are embedded in the model weights, either locally through
`LocalHFBackend` or through an OpenAI-compatible vLLM deployment. Only adapter
functions listed in a model's `adapter_index.json` are available. For setup and
backend details, see [Adapter functions](../../docs/advanced/intrinsics.md) and the
[official Granite Switch documentation](https://github.com/generative-computing/granite-switch).
For runtime LoRA/aLoRA adapters on a standard backend, see
[`intrinsics/`](../intrinsics/README.md).

## Examples

| File | Description |
| :--- | :--- |
| [`answerability_local_hf.py`](./answerability_local_hf.py) | `rag.check_answerability()` on a local Granite Switch checkpoint using `LocalHFBackend(load_embedded_adapters=True)`. Needs a GPU or Apple Silicon Mac (`uv sync --extra hf`). |
| [`answerability_openai.py`](./answerability_openai.py) | `rag.check_answerability()` on a vLLM-served Granite Switch model using `OpenAIBackend(load_embedded_adapters=True)`. Needs a running vLLM server (`uv sync --extra switch`). |
| [`hallucination_detection_openai.py`](./hallucination_detection_openai.py) | `rag.flag_hallucinated_content()` on a vLLM-served Granite Switch model using `OpenAIBackend(load_embedded_adapters=True)`. Needs a running vLLM server (`uv sync --extra switch`). |
| [`manual_adapter_loading.py`](./manual_adapter_loading.py) | Registering a subset of embedded adapters with `register_embedded_adapter_model(intrinsic_name=...)` instead of loading all of them. Needs a running vLLM server (`uv sync --extra switch`). |
