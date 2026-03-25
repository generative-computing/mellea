---
name: estimate-vram
description: >
  Estimate GPU VRAM and RAM requirements for test files by tracing model
  identifiers and looking up parameter counts. Use when migrating legacy
  requires_gpu/requires_heavy_ram markers to predicates, or adding resource
  gating to new GPU tests. Produces a recommendations table with exact
  require_gpu(min_vram_gb=N) and require_ram(min_gb=N) values.
argument-hint: "[file-or-directory] [--precision fp16|fp32|int4|int8]"
compatibility: "Claude Code, IBM Bob"
metadata:
  version: "2026-03-25"
  capabilities: [read_file, bash, grep, glob]
---

# Estimate VRAM Requirements

Analyse test files to determine appropriate `require_gpu(min_vram_gb=N)` and
`require_ram(min_gb=N)` values for resource gating predicates.

## Inputs

- `$ARGUMENTS` — file path, directory, or glob. If empty, scan all test files
  that have GPU-related markers or backend constructors.
- `--precision` — override default precision assumption. One of `fp16` (default),
  `fp32`, `int4`, `int8`.

## Project References

Read these before estimating — they provide model constants and predicate APIs:

- **Model identifiers:** `mellea/backends/model_ids.py` (`ModelIdentifier` constants)
- **Resource predicates:** `test/predicates.py` (available predicate functions)
- **Marker conventions:** `test/MARKERS_GUIDE.md`

---

## Procedure

### Step 1 — Find GPU-relevant test files

If `$ARGUMENTS` is a specific file, use that. Otherwise grep across the target
scope for any of these signals:

- `requires_gpu`, `require_gpu` — existing or legacy resource gating
- `requires_heavy_ram`, `require_ram` — existing or legacy RAM gating
- `LocalHFBackend`, `LocalVLLMBackend` — local GPU backends
- `.from_pretrained(` — direct model loading
- `pytest.mark.huggingface`, `pytest.mark.vllm` — GPU backend markers

Files with only `pytest.mark.ollama` or cloud backend markers (openai, watsonx,
litellm, bedrock) do not need GPU gating analysis — skip them unless they also
have a `requires_gpu` marker that may be wrong (like `test_genslot.py`).

### Step 2 — Trace the model identifier

For each file, determine which model(s) it loads. Check in order:

1. **Module-level constants** — e.g. `BASE_MODEL = "ibm-granite/..."` or
   `MODEL_ID = model_ids.QWEN3_0_6B`.
2. **Fixture definitions** — trace `@pytest.fixture` functions for:
   - `LocalHFBackend(model_id=...)` — extract the `model_id` argument
   - `LocalVLLMBackend(model_id=...)` — extract the `model_id` argument
   - `start_session("hf", model_id=...)` — extract `model_id`
3. **ModelIdentifier resolution** — if the model_id is a constant like
   `model_ids.QWEN3_0_6B`, read `mellea/backends/model_ids.py` and extract
   the `hf_model_name` field.
4. **Conftest fixtures** — check `conftest.py` files up the directory tree for
   fixture definitions that provide model/backend instances.
5. **Per-function overrides** — some files have different models per test function.
   Track per-function when this occurs.

Record the model ID as a HuggingFace repo name (e.g. `"ibm-granite/granite-4.0-micro"`)
or an Ollama tag (e.g. `"granite4:micro-h"`) — whichever is available.

### Step 3 — Look up parameter count

Use three strategies in priority order. Stop at the first that succeeds.

#### Strategy A: HuggingFace Hub API (preferred, requires network)

If the model ID is an HF repo name and `huggingface_hub` is available:

```python
from huggingface_hub.utils._safetensors import get_safetensors_metadata
meta = get_safetensors_metadata("ibm-granite/granite-3.3-8b-instruct")
total_params = sum(meta.parameter_count.values())
```

This returns exact parameter counts by dtype. Use it when available.

Run this via `uv run python -c "..."` — the agent does not need torch installed,
only `huggingface_hub` (which is in the `[hf]` extra).

If `huggingface_hub` is not installed or network is unavailable, fall through.

#### Strategy B: Ollama model info (for Ollama-tagged models)

If the model has an Ollama name and Ollama is running:

```bash
ollama show <model_name> --modelfile 2>/dev/null | grep -i 'parameter'
```

Or parse the Ollama tag for size hints (see Strategy C).

#### Strategy C: Model name parsing (offline fallback)

Extract parameter count from common naming patterns in the model ID string.
Match against these regex patterns (case-insensitive):

| Pattern | Extract | Example match |
|---------|---------|---------------|
| `(\d+\.?\d*)b[-_.]` or `-(\d+\.?\d*)b` | N billion | `granite-3.3-8b` → 8B |
| `(\d+\.?\d*)B` (capital B in HF names) | N billion | `Qwen3-0.6B` → 0.6B |
| `-(\d+)m[-_.]` or `(\d+)m-` | N million ÷ 1000 | `granite-4.0-h-350m` → 0.35B |
| `micro` without explicit size | 0.35B–3B | Check ModelIdentifier catalog |
| `tiny` without explicit size | 1B–7B | Check ModelIdentifier catalog |
| `small` without explicit size | 3B–8B | Check ModelIdentifier catalog |

When the name is ambiguous (e.g. `granite4:micro-h` has no explicit number),
resolve via the `ModelIdentifier` constant in `model_ids.py` — the HF name
usually contains the explicit size.

#### Strategy D: Conservative default (last resort)

If the model cannot be identified after A–C:
- Assume **8B parameters** (16 GB at fp16)
- Flag the file as **"model unidentified — manual review needed"**

### Step 4 — Determine backend type

The backend determines whether GPU gating is needed at all:

| Backend | GPU loaded locally? | Predicate needed |
|---------|--------------------|--------------------|
| `LocalHFBackend` | Yes | `require_gpu(min_vram_gb=N)` |
| `LocalVLLMBackend` | Yes | `require_gpu(min_vram_gb=N)` |
| `OllamaModelBackend` | Managed by Ollama | `require_ollama()` only. Exception: models >8B through Ollama may need `require_ram(min_gb=N)` for the Ollama server process. |
| `OpenAIBackend` (real API) | No | No GPU gate |
| `OpenAIBackend` → Ollama `/v1` | Managed by Ollama | `require_ollama()` only |
| `WatsonxAIBackend` | No | No GPU gate |
| `LiteLLMBackend` | No | No GPU gate |
| Cloud API (Bedrock, etc.) | No | No GPU gate |

**Key rule:** Ollama manages its own GPU memory. Tests using Ollama backends
should use `require_ollama()`, NOT `require_gpu()`. The only exception is when
a test needs guaranteed GPU performance (rare) or uses a very large model where
insufficient system RAM would cause Ollama to fail.

### Step 5 — Compute VRAM and RAM estimates

#### VRAM formula

```
vram_gb = params_B × bytes_per_param × 1.2
```

Where:
- `params_B` = parameter count in billions
- `bytes_per_param` depends on precision:
  - fp32: 4.0
  - fp16/bf16: 2.0 (default assumption)
  - int8: 1.0
  - int4: 0.5
- `1.2` = 20% overhead for KV cache, activations, framework buffers

Round `min_vram_gb` **up** to the next even integer.

#### RAM formula

For local GPU backends (HF, vLLM):
```
min_ram_gb = max(16, vram_gb + 8)
```

The `+ 8` accounts for OS, Python runtime, data loading, and test framework.
Minimum 16 GB because the test environment needs a working OS + IDE.

For Ollama backends with large models (>8B):
```
min_ram_gb = max(16, vram_gb + 12)
```

The `+ 12` accounts for the Ollama server process overhead on top of the model.

#### GPU isolation

If a test uses `LocalHFBackend` or `LocalVLLMBackend`, recommend
`require_gpu_isolation()` in addition to `require_gpu()`. These backends hold
GPU memory at the process level and need subprocess isolation for multi-module
test runs.

### Step 6 — Output recommendations

#### Summary table (always print first)

```
| File | Model | Params | Backend | VRAM (GB) | Recommended predicates |
|------|-------|--------|---------|-----------|------------------------|
```

Each row should show:
- File path (relative to repo root)
- Model identifier (short form)
- Parameter count (e.g. "8B", "350M")
- Backend type (HF, vLLM, Ollama, API)
- Computed VRAM at the selected precision
- The exact predicate call(s) to use

#### Flag categories

| Flag | Meaning |
|------|---------|
| `model unidentified` | Strategies A–C all failed. Default 8B applied. Manual review needed. |
| `remove GPU gate` | Test uses Ollama/API backend — `require_gpu()` is unnecessary. |
| `multi-GPU required` | VRAM exceeds 48 GB — cannot run on single consumer GPU. |
| `verify precision` | Default fp16 assumed but test may use quantisation. |

#### Footer

```
---
Files analysed: N | Estimates computed: N | Manual review needed: N
Precision: fp16 (default) | Override with --precision
```

---

## Scope boundaries

This skill does NOT:
- Modify test files (it only produces recommendations)
- Run the actual models
- Modify `predicates.py` or `conftest.py`
- Determine tier classification (use `/audit-markers` for that)

The output feeds into `/audit-markers` when migrating legacy resource markers
to predicates.
