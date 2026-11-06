# m decompose

`m decompose` helps you break a complex task prompt into structured, dependency-aware subtasks.

The decomposition pipeline extracts constraints, generates prompt templates for each subtask, and writes runnable outputs so you can inspect and execute the workflow.

For a full guide, see [m decompose](../../docs/how-to/m-decompose.md); for every
option, see the [CLI reference](../../docs/reference/cli.md).

## Examples

| File | Description |
| :--- | :--- |
| [`python/python_decompose_example.py`](./python/python_decompose_example.py) | Calls the `decompose` pipeline directly from Python: breaks down a blog-post task, saves the results as JSON, renders an executable Python script from the decomposition template, and optionally runs it. |
| [`decompose_using_cli.sh`](./decompose_using_cli.sh) | Minimal `m decompose run` invocation: writes a task to a file, creates the output directory, and runs the CLI. |
| [`decompose_intermediary_python_command.sh`](./decompose_intermediary_python_command.sh) | Runs `m decompose run` and then executes the generated Python script to produce the final output. |

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
