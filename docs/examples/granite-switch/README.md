# Granite Switch Examples

This directory contains examples for running Mellea adapter functions with
Granite Switch models, either locally through `LocalHFBackend` or through an
OpenAI-compatible vLLM or llama.cpp server. For setup and backend details, see
[Adapter functions](../../docs/advanced/intrinsics.md) and the
[official Granite Switch documentation](https://github.com/generative-computing/granite-switch).
For runtime LoRA/aLoRA adapters on a standard backend, see
[`intrinsics/`](../intrinsics/README.md).

Granite Switch models ship with LoRA and aLoRA adapters pre-baked into the model
weights. Unlike runtime LoRA/aLoRA adapters on a standard `LocalHFBackend`,
these embedded adapters are activated via control tokens injected by the model's
chat template. Only the I/O transformation configs are downloaded — no adapter
weights are transferred.

Not all adapter functions are embedded in every Granite Switch model. Check the
model's `adapter_index.json` for a definitive list. For Granite Switch models
pre-built by IBM, Mellea includes a list of models in `model_id`.

## Examples

| File | Description |
| :--- | :--- |
| [`answerability_local_hf.py`](./answerability_local_hf.py) | Demonstrates `rag.check_answerability()` against a local Granite Switch checkpoint using `LocalHFBackend(load_embedded_adapters=True)`. Needs a GPU or Apple Silicon Mac (`uv sync --extra hf`). |
| [`answerability_openai.py`](./answerability_openai.py) | Demonstrates `rag.check_answerability()` using `OpenAIBackend` with `load_embedded_adapters=True` — the simplest way to use adapter functions with Granite Switch. Needs a running vLLM server (`uv sync --extra switch`). |
| [`answerability_llamacpp.py`](./answerability_llamacpp.py) | Demonstrates `rag.check_answerability()` against a local llama-server started by [`scripts/start_llamacpp_switch.sh`](../../../scripts/start_llamacpp_switch.sh). Skips with an actionable reason when that server isn't running (see [Running on llama.cpp](#running-on-llamacpp)). |
| [`hallucination_detection_openai.py`](./hallucination_detection_openai.py) | Demonstrates `rag.flag_hallucinated_content()` using `OpenAIBackend` with `load_embedded_adapters=True`. Needs a running vLLM server (`uv sync --extra switch`). |
| [`manual_adapter_loading.py`](./manual_adapter_loading.py) | Shows how to manually load embedded adapters using `backend.register_embedded_adapter_model(intrinsic_name=...)`. Useful when you only need a subset of adapters or want more control over adapter registration. Needs a running vLLM server (`uv sync --extra switch`). |

## Running on llama.cpp

`answerability_llamacpp.py` needs a local [llama.cpp](https://github.com/ggml-org/llama.cpp)
server (v0.1.0 or newer, which carries the granite-switch architecture; tested
on v0.6.0). Start one with:

```bash
./scripts/start_llamacpp_switch.sh
```

The first run downloads the checkpoint from the Hugging Face Hub (~7 GB for the
3b) and converts it to GGUF (~9 GB more; both cached under
`~/.cache/mellea/llamacpp-switch-build`). Later runs reuse the cache and start
in seconds. The server listens on `http://127.0.0.1:8081`, deliberately not the
base-Granite server's 8080
([`scripts/start_llamacpp.sh`](../../../scripts/start_llamacpp.sh)), so both can
run side by side; `LLAMACPP_SWITCH_PORT` changes it.

The example runs whenever that server is reachable (`LLAMACPP_SWITCH_BASE_URL`,
default `http://127.0.0.1:8081/v1`) and serves the Switch model, and skips with
an actionable reason otherwise. It never downloads model weights; its first run
does fetch the small adapter configuration files (`adapter_index.json` and the
`io_configs/`) from the Hub, which are cached.

## Architecture

![Granite Libraries Software Stack Architecture in Mellea](../../static/images/granite-libraries-mellea-architecture.png)
