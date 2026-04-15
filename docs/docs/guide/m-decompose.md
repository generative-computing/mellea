---
title: "m decompose"
description: "Break complex tasks into ordered, executable subtasks with the m decompose CLI."
# diataxis: how-to
---

`m decompose` takes a complex task description and uses an LLM to:

1. Extract the constraints the output must satisfy
2. Identify the subtasks needed to complete the goal, with dependency ordering
3. Generate a prompt template for each subtask
4. Output a ready-to-run Python script that executes each subtask in order

**Prerequisites:** Mellea installed (`uv add "mellea[cli]"`), Ollama running locally (or an OpenAI-compatible endpoint).

## Basic usage

Write your task description to a text file, then run:

```bash
mkdir -p ./output
m decompose run --input-file task.txt --out-dir ./output/
```

> **Note:** The output directory must already exist — the command will error if it
> does not. On first run with Ollama, the default model will be downloaded
> automatically (~15 GB for the full model). Use `--model-id` with a smaller model
> (e.g. `granite4:micro`) to avoid the large download.

This produces a subdirectory under `./output/` (one per task job):

- `./output/m_decomp_result/m_decomp_result.json` — the full decomposition:
  subtask list, constraints, dependency graph, and prompt templates
- `./output/m_decomp_result/m_decomp_result.py` — a runnable Python script that calls
  `m.instruct()` for each subtask in dependency order

## Example

Given a `task.txt`:

```text
Write a short blog post about the benefits of morning exercise.
Include a catchy title, an introduction paragraph, three main benefits
with explanations, and a conclusion that encourages readers to start
their morning exercise routine.
```

Run:

```bash
m decompose run --input-file task.txt --out-dir ./output/
```

Then execute the generated script:

```bash
python output/m_decomp_result/m_decomp_result.py
```

## Backend options

`m decompose` defaults to Ollama with `granite4:micro`. Pass `--backend` and
`--model-id` to use a different inference engine:

```bash
m decompose run \
  --input-file task.txt \
  --out-dir ./output/ \
  --backend openai \
  --model-id gpt-4o-mini
```

To see all options:

```bash
m decompose --help
m decompose run --help
```

## Python API

Use the decompose pipeline directly from Python:

```python
from cli.decompose.pipeline import DecompBackend, decompose

result = decompose(
    task_prompt="Write a short blog post about morning exercise.",
    model_id="granite4:micro",
    backend=DecompBackend.ollama,
)

# result["subtask_list"]       — ordered list of subtask descriptions
# result["identified_constraints"] — constraints extracted from the prompt
# result["subtasks"]           — detailed subtask objects with prompt templates
```

Each subtask in `result["subtasks"]` has:

| Field | Description |
| --- | --- |
| `subtask` | Description of the subtask |
| `tag` | Short identifier used for dependency references |
| `depends_on` | List of `tag` values this subtask depends on |
| `prompt_template` | Ready-to-use prompt string for `m.instruct()` |
| `input_vars_required` | Variables that must be filled in the template |
| `constraints` | Constraints from the original prompt that apply here |

## When to use m decompose

`m decompose` is useful when:

- A task prompt is too large or complex for a single LLM call
- The work can be broken into sequential or parallel subtasks
- You want a first-pass structure you can then edit by hand
- You are exploring how to decompose a problem before writing code

For tasks that fit comfortably in a single prompt, use `m.instruct()` directly.

---

**Full example:** [`docs/examples/m_decompose/`](https://github.com/generative-computing/mellea/blob/main/docs/examples/m_decompose/)

---

**See also:** [Tools and Agents](../guide/tools-and-agents) | [Refactor Prompts with CLI](../how-to/refactor-prompts-with-cli) | [CLI Reference](../reference/cli)
