---
title: "CLI Reference"
sidebarTitle: "CLI Reference"
description: "Complete reference for the m command-line tool — all subcommands, flags, and defaults."
---

Mellea command-line tool for LLM-powered workflows.

Provides sub-commands for serving models (`m serve`), training and uploading
adapters (`m alora`), decomposing tasks into subtasks (`m decompose`),
running test-based evaluation pipelines (`m eval`), and applying automated
code migrations (`m fix`).

## `m alora`

Train or upload aLoRAs for requirement validation.

### `m alora add-readme`

Generate and upload an INTRINSIC_README.md for a trained adapter.

Uses an LLM to auto-generate documentation for a trained adapter based on
the training data and model configuration, then uploads it to the Hugging
Face Hub repository.

> **Prerequisites:**
>
> - Hugging Face CLI authenticated (`huggingface-cli login`).
> - An LLM backend available for README generation.

```bash
m alora add-readme <DATAFILE> --basemodel <value> [--promptfile] --name <value> [--hints] [--io-yaml]
```

**Arguments:**

| Name | Type | Required | Description |
| ---- | ---- | -------- | ----------- |
| `DATAFILE` | text | yes | JSONL file with item/label pairs |

**Options:**

| Flag | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `--basemodel` | text | *required* | Base model ID or path |
| `--promptfile` | text | — | Path to load the prompt format file |
| `--name` | text | *required* | Destination model name (e.g., acme/carbchecker-alora) |
| `--hints` | text | — | File containing any additional hints. |
| `--io-yaml` | text | — | Location of the io.yaml file that configures input and output processing if the model is invoked as an intrinsic. |

**Output:** Generates a README.md file, displays it for confirmation, and uploads it to the Hugging Face Hub repository specified by `--name`.

**Example:**

```bash
m alora add-readme data.jsonl --basemodel ibm-granite/granite-3.3-2b-instruct --name acme/my-alora
```

**See also:** [Lora and Alora Adapters](../advanced/lora-and-alora-adapters)

### `m alora train`

Train an aLoRA or LoRA adapter on a labelled dataset.

Fine-tunes a base causal language model using a JSONL dataset of item/label
pairs. Supports both aLoRA (asymmetric LoRA) and standard LoRA adapters.

> **Prerequisites:**
>
> - Mellea installed with adapter extras (`uv add mellea[adapters]`).
> - A CUDA, MPS, or CPU device available for training.

```bash
m alora train <DATAFILE> --basemodel <value> --outfile <value> [--promptfile] [--adapter] [--device] [--epochs] [--learning-rate] [--batch-size] [--max-length] [--grad-accum]
```

**Arguments:**

| Name | Type | Required | Description |
| ---- | ---- | -------- | ----------- |
| `DATAFILE` | text | yes | JSONL file with item/label pairs |

**Options:**

| Flag | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `--basemodel` | text | *required* | Base model ID or path |
| `--outfile` | text | *required* | Path to save adapter weights |
| `--promptfile` | text | — | Path to load the prompt format file |
| `--adapter` | text | `alora` | Adapter type: alora or lora |
| `--device` | text | `auto` | Device: auto, cpu, cuda, or mps |
| `--epochs` | integer | `6` | Number of training epochs |
| `--learning-rate` | float | `6e-06` | Learning rate |
| `--batch-size` | integer | `2` | Per-device batch size |
| `--max-length` | integer | `1024` | Max sequence length |
| `--grad-accum` | integer | `4` | Gradient accumulation steps |

**Output:** Saves adapter weights to the path specified by `--outfile`. The output directory contains an `adapter_config.json` and the trained weight files, ready for upload or local inference.

**Example:**

```bash
m alora train data.jsonl --basemodel ibm-granite/granite-3.3-2b-instruct --outfile ./adapter
```

**See also:** [Lora and Alora Adapters](../advanced/lora-and-alora-adapters)

### `m alora upload`

Upload a trained adapter to a remote model registry.

Pushes adapter weights to Hugging Face Hub, optionally packaging the adapter
as an intrinsic with an `io.yaml` configuration file.

> **Prerequisites:** Hugging Face CLI authenticated (`huggingface-cli login`).

```bash
m alora upload <WEIGHT_PATH> --name <value> [--intrinsic] [--io-yaml]
```

**Arguments:**

| Name | Type | Required | Description |
| ---- | ---- | -------- | ----------- |
| `WEIGHT_PATH` | text | yes | Path to saved adapter weights |

**Options:**

| Flag | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `--name` | text | *required* | Destination model name (e.g., acme/carbchecker-alora) |
| `--intrinsic` | boolean | `false` | True if the uploaded adapter implements an intrinsic. If true, the caller must provide an io.yaml file. |
| `--io-yaml` | text | — | Location of the io.yaml file that configures input and output processing if the model is invoked as an intrinsic. |

**Output:** Creates or updates a Hugging Face Hub repository at the name specified by `--name` and uploads the adapter weight files.

**Example:**

```bash
m alora upload ./adapter --name acme/my-alora
```

**See also:** [Lora and Alora Adapters](../advanced/lora-and-alora-adapters)

## `m decompose`

Utility pipeline for decomposing task prompts.

### `m decompose run`

Break a complex task into ordered, executable subtasks.

Reads user queries from a file or interactive input, runs the LLM-driven
decomposition pipeline for each task job, and writes one JSON file, one
rendered Python script, and any generated validation modules under a per-job
output directory.

> **Prerequisites:**
>
> - Mellea installed (`uv add mellea`).
> - An Ollama instance running locally, or an OpenAI-compatible endpoint configured via `--backend-endpoint`.

```bash
m decompose run --out-dir <value> [--out-name] [--input-file] [--model-id] [--backend] [--backend-req-timeout] [--backend-endpoint] [--backend-api-key] [--version] [--input-var] [--log-mode] [--enable-script-run]
```

**Options:**

| Flag | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `--out-dir` | path | *required* | Path to an existing directory to save the output files. |
| `--out-name` | text | `m_decomp_result` | Name for the output files. Defaults to "m_decomp_result". |
| `--input-file` | text | — | Path to a text file containing user queries. |
| `--model-id` | text | `mistral-small3.2:latest` | Model name/id used to run the decomposition pipeline. Defaults to "mistral-small3.2:latest", valid for the "ollama" backend. |
| `--backend` | ollama \| openai | `ollama` | Backend used for inference. Options: "ollama" and "openai". |
| `--backend-req-timeout` | integer | `300` | Timeout in seconds for backend requests. Defaults to "300". |
| `--backend-endpoint` | text | — | Backend endpoint / base URL. Required for "openai". |
| `--backend-api-key` | text | — | Backend API key. Required for "openai". |
| `--version` | latest \| v1 \| v2 \| v3 | `latest` | Version of the mellea program generator template to use. |
| `--input-var` | text | — | Optional user input variable names. You may pass this option multiple times. Each value must be a valid Python identifier. |
| `--log-mode` | demo \| debug | `demo` | Readable logging mode. Options: "demo" or "debug". |
| `--enable-script-run` | boolean | `false` | When true, generated scripts expose argparse runtime options for backend, model, endpoint, and API key overrides. |

**Output:** Creates a directory `<out-dir>/<out-name>/` containing a JSON decomposition result file, a ready-to-run Python script, and any generated validation modules. One directory per task job.

**Example:**

```bash
m decompose run --out-dir ./output --input-file tasks.txt
```

**See also:** [M Decompose](../guide/m-decompose), [Refactor Prompts with Cli](../how-to/refactor-prompts-with-cli)

## `m eval`

LLM-as-a-judge evaluation pipelines.

### `m eval run`

Run LLM-as-a-judge evaluation on one or more test files.

Loads test cases from JSON/JSONL files, generates candidate responses using
the specified generation backend, scores them with a judge model, and writes
aggregated results to a file.

> **Prerequisites:**
>
> - Mellea installed (`uv add mellea`).
> - At least one inference backend available (Ollama by default).
> - A separate judge backend/model is recommended but optional (defaults to the generation backend).

```bash
m eval run <TEST_FILES> [--backend] [--model] [--max-gen-tokens] [--judge-backend] [--judge-model] [--max-judge-tokens] [--output-path] [--output-format] [--continue-on-error]
```

**Arguments:**

| Name | Type | Required | Description |
| ---- | ---- | -------- | ----------- |
| `TEST_FILES` | text | yes | List of paths to json/jsonl files containing test cases |

**Options:**

| Flag | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `--backend`, `-b` | text | `ollama` | Inference backend for generating candidate responses (e.g. ollama, openai) |
| `--model` | text | — | Model name/id for the generation backend; uses backend default if omitted |
| `--max-gen-tokens` | integer | `256` | Max tokens to generate for responses |
| `--judge-backend`, `-jb` | text | — | Inference backend for the judge model; reuses --backend if omitted |
| `--judge-model` | text | — | Model name/id for the judge; uses judge backend default if omitted |
| `--max-judge-tokens` | integer | `256` | Max tokens for the judge model's judgement. |
| `--output-path`, `-o` | text | `eval_results` | Output path for results |
| `--output-format` | text | `json` | Either json or jsonl format for results |
| `--continue-on-error` | boolean | `true` | Skip failed test cases instead of aborting the entire run |

**Output:** Writes evaluation results to `<output-path>.<output-format>` (default `eval_results.json`). The file contains per-test-case scores, judge verdicts, and aggregate statistics.

**Example:**

```bash
m eval run tests.jsonl --backend ollama --model granite3.3:2b
```

**See also:** [Evaluate with Llm as a Judge](../evaluation-and-observability/evaluate-with-llm-as-a-judge)

## `m fix`

Fix code for API changes.

### `m fix async`

Fix async calls for the await_result default change.

Scans Python source files for `aact`, `ainstruct`, and `aquery` calls
and applies an automated migration to restore blocking behaviour after the
`await_result` default changed from `True` to `False`.

> **Prerequisites:** Mellea installed (`uv add mellea`).

```bash
m fix async <PATH> [--mode] [--dry-run]
```

**Arguments:**

| Name | Type | Required | Description |
| ---- | ---- | -------- | ----------- |
| `PATH` | text | yes | File or directory to scan |

**Options:**

| Flag | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `--mode`, `-m` | add-await-result \| add-stream-loop | `add-await-result` | Fix strategy to apply |
| `--dry-run` | boolean | `false` | Report locations without modifying files |

**Modes:**

- **`add-await-result`** — (default) Adds await_result=True to each call so it blocks until the result is ready. Use this if you don't need to stream partial results.
- **`add-stream-loop`** — Inserts a `while not r.is_computed(): await r.astream()` loop after each call. This only works if you passed a streaming model option (e.g. stream=True) to the call; otherwise the loop will finish immediately.

**Best practices:**

- Run with --dry-run first to review what will be changed.
- Only run a given mode once per file. The tool detects prior fixes and skips calls that already have await_result=True or a stream loop, but it is safest to treat it as a one-shot migration.
- Do not run both modes on the same file. If a stream loop is already present, add-await-result will skip that call (and vice versa).

**Detection notes:**

- Most import styles are detected: `import mellea`, `from mellea import MelleaSession`, `from mellea.stdlib.functional import aact`, module aliases, etc.
- Calls that are already followed by `await r.avalue()`, `await r.astream()`, or a `while not r.is_computed()` loop are automatically skipped, even when nested inside if/try/for blocks.

**Output:** Modifies Python source files in place (unless `--dry-run`). Prints a summary of fixed call sites with file paths and line numbers.

**Example:**

```bash
m fix async src/ --dry-run
```

### `m fix genslots`

Rewrite genslot imports and class names to genstub equivalents.

Scans Python source files and replaces deprecated `GenerativeSlot` imports
and class references with their `GenerativeStub` replacements.

> **Prerequisites:** Mellea installed (`uv add mellea`).

```bash
m fix genslots <PATH> [--dry-run]
```

**Arguments:**

| Name | Type | Required | Description |
| ---- | ---- | -------- | ----------- |
| `PATH` | text | yes | File or directory to scan |

**Options:**

| Flag | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `--dry-run` | boolean | `false` | Report locations without modifying files |

**Rewrites:**

- mellea.stdlib.components.genslot → mellea.stdlib.components.genstub
- GenerativeSlot      → GenerativeStub
- SyncGenerativeSlot  → SyncGenerativeStub
- AsyncGenerativeSlot → AsyncGenerativeStub

**Best practices:**

- Run with --dry-run first to review what will be changed.
- The tool is idempotent — running it twice on the same file is safe.

**Output:** Modifies Python source files in place (unless `--dry-run`). Prints a summary of rewritten references with file paths and line numbers.

**Example:**

```bash
m fix genslots src/ --dry-run
```

## `m serve`

Serve a Mellea program as an OpenAI-compatible HTTP endpoint.

Loads a Python file containing a `serve` function and exposes it
via a FastAPI server implementing the OpenAI chat completions API. The server
accepts `POST /v1/chat/completions` requests.

> **Prerequisites:**
>
> - Mellea installed with server dependency group (`uv add 'mellea[server]'`).
> - The python file being loaded must have a `serve` function.

```bash
m serve [SCRIPT_PATH] [--host] [--port]
```

**Arguments:**

| Name | Type | Required | Description |
| ---- | ---- | -------- | ----------- |
| `SCRIPT_PATH` | text | no | Path to the Python script to import and serve |

**Options:**

| Flag | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `--host` | text | `0.0.0.0` | Host to bind to |
| `--port` | integer | `8080` | Port to bind to |

**Output:** Starts a long-running HTTP server on the specified host and port. The `/v1/chat/completions` endpoint accepts OpenAI-format chat completion requests and returns `ChatCompletion` JSON responses.

**Example:**

```bash
m serve my_app.py --port 9000
```

**See also:** [M Serve](../integrations/m-serve)
