# m decompose

`m decompose` helps you break a complex task prompt into structured, dependency-aware subtasks.

The decomposition pipeline extracts constraints, generates prompt templates for each subtask, and writes runnable outputs so you can inspect and execute the workflow.

## How users run it (current behavior)

### CLI mode (recommended)

1. Prepare an output directory (must already exist).
2. Put task prompt(s) in a text file.
3. Run `m decompose run`.

```bash
MODEL_ID=mistral-small3.2:latest  # e.g. granite4:latest

mkdir -p ./output

m decompose run \
  --model-id $MODEL_ID \
  --input-file task.txt \
  --out-dir ./output \
  --out-name my_decomp
```

Important runtime behavior:

- `--input-file` supports **multiple non-empty lines**. Each line is treated as one task job.
- Multiple jobs produce numbered outputs: `my_decomp_1/`, `my_decomp_2/`, ...
- Outputs are written under `out_dir/out_name/` (or numbered job directories).
- Backend default: `ollama`
- Model default: `mistral-small3.2:latest`

### Interactive mode

If `--input-file` is omitted, the CLI prompts for one task string interactively.

Note:

- Interactive mode is intended for single prompt input.
- `--input-var` is ignored in interactive mode by current implementation.

## Output structure

For one query, `m decompose run` creates:

```text
<out-dir>/<out-name>/
├── <out-name>.json
├── <out-name>.py
└── validations/
    ├── __init__.py
    └── val_fn_*.py  # only when a constraint uses code validation
```
For multiple queries:
```text
<out-dir>/
├── <out-name>_1/
├── <out-name>_2/
└── ...
```

- `*.json`: full decomposition result (`subtask_list`, `identified_constraints`, `subtasks`, ...)
- `*.py`: rendered runnable program from the selected template version (`latest` currently resolves to `v2`)
- `validations/`: generated validation helper functions for constraints using `code` strategy



## Key CLI options

- `--backend`: `ollama` | `openai` | `rits`
- `--model-id`: inference model id/name
- `--backend-endpoint`: required for `openai` and `rits`
- `--backend-api-key`: required for `openai` and `rits`
- `--backend-req-timeout`: request timeout (seconds), default `300`
- `--input-var`: optional input variable names (repeatable, must be valid Python identifiers)
- `--version`: template version (`latest`, `v1`, `v2`)
- `--log-mode`: `demo` | `debug`

## Python API (pipeline interface)

You can call the decomposition pipeline directly:

```python
import json

from cli.decompose.pipeline import DecompBackend, decompose

result = decompose(
    task_prompt="Write a short blog post about morning exercise.",
    user_input_variable=["USER_CONTEXT"],
    model_id="mistral-small3.2:latest",
    backend=DecompBackend.ollama,
)

print(json.dumps(result, indent=2, ensure_ascii=False))
```

`result["subtasks"]` items include:

- `subtask`
- `tag`
- `prompt_template`
- `general_instructions`
- `input_vars_required`
- `depends_on`
- `constraints` (with `val_strategy`, `val_fn_name`, `val_fn`)

## Example: OpenAI-compatible endpoint

```bash
m decompose run \
  --input-file task.txt \
  --out-dir ./output \
  --backend openai \
  --model-id gpt-4o-mini \
  --backend-endpoint http://localhost:8000/v1 \
  --backend-api-key EMPTY
```

## What this example demonstrates

- Decomposing one large task into manageable subtasks
- Preserving explicit constraints across subtasks
- Producing inspectable intermediate artifacts for debugging and editing
- Generating a runnable decomposition program instead of a single opaque prompt call
