---
canonical: "https://docs.mellea.ai/how-to/refactor-prompts-with-cli"
title: "Refactor Prompts with the CLI"
sidebarTitle: "Refactor with m decompose"
description: "Use m decompose to break a complex prompt into typed, validated generative functions."
# diataxis: how-to
---

**Prerequisites:** `pip install "mellea[cli]"`, Ollama running locally (or an
OpenAI-compatible endpoint).

When a single prompt grows too long or asks the LLM to do too many things at
once, quality degrades. `m decompose` analyses the prompt, extracts its
constraints, and produces a Python script of ordered `m.instruct()` calls — one
per subtask — that you can run immediately or refine with types and requirements.

---

## When to use m decompose

Use `m decompose` when:

- A prompt contains multiple distinct tasks (write, classify, translate,
  summarise) that you would benefit from separating.
- You want to add typed return values or `@generative` wrappers to each step.
- You need to assign different requirements to different parts of the pipeline.
- You are prototyping a pipeline and want a structured starting point to edit.

For prompts that fit cleanly in a single `m.instruct()` call, use `instruct()`
directly.

---

## Step 1: Write your prompt to a file

Create a plain-text file that describes the full task. Include all constraints
and requirements as part of the description — `m decompose` extracts them:

```text
Plan a birthday party for a 10-year-old.

The plan must include:
- A theme suggestion with a short explanation
- A list of at least 5 activities suitable for children aged 8-12
- A catering menu with a main dish, two sides, and a birthday cake option
- A 30-word invitation message addressed to the child's classmates

All content must be age-appropriate. The invitation must not exceed 30 words.
The activity list must be ordered from most energetic to least energetic.
```

Save this as `party_plan.txt`.

> **Tip:** The more explicit your constraints in the prompt file, the more
> accurately `m decompose` assigns them to individual subtasks. Phrases like
> "must", "must not", "at least", and "ordered by" are reliably extracted as
> constraints.

---

## Step 2: Run the decompose command

```bash
m decompose run --prompt-file party_plan.txt --out-dir ./output/
```

This produces two files in `./output/`:

- `m_decomp_result.py` — a runnable Python script with one `m.instruct()` call
  per subtask, in dependency order
- `m_decomp_result.json` — the full decomposition: subtask list, extracted
  constraints, dependency graph, and Jinja2 prompt templates

> **Note:** The `--out-dir` directory must already exist. `m decompose` does not
> create it.

### What the pipeline does

`m decompose` runs these steps internally, in order:

1. Parses the prompt into a list of subtasks, each tagged with a short
   identifier
2. Extracts all constraints and requirements from the prompt text
3. Decides for each constraint whether validation should be done with code
   (`"code"`) or with an LLM judge (`"llm"`)
4. Generates a Jinja2 prompt template for each subtask
5. Assigns constraints to the subtasks they apply to
6. Writes the output Python script with `m.instruct()` calls in dependency order

### All CLI options

| Flag | Default | Description |
| --- | --- | --- |
| `--prompt-file` | (interactive) | Path to a text file containing the task prompt. Omit to enter the prompt interactively. |
| `--out-dir` | (required) | Path to the directory for output files. Must exist. |
| `--out-name` | `m_decomp_result` | Base name for the output `.py` and `.json` files. |
| `--model-id` | `mistral-small3.2:latest` | Model to use for the decomposition. |
| `--backend` | `ollama` | Inference backend: `ollama` or `openai`. |
| `--backend-endpoint` | — | URL endpoint. Required when `--backend openai`. |
| `--backend-api-key` | — | API key. Required when `--backend openai`. |
| `--backend-req-timeout` | `300` | Request timeout in seconds. |
| `--input-var` | — | Repeatable. Declares a user input variable name (uppercase Python identifier). |

---

## Step 3: Review the generated Python file

Open `output/m_decomp_result.py`. For the birthday party prompt above, the
generated script looks roughly like this:

```python
import textwrap
import mellea

m = mellea.start_session()

# Subtask: suggest_theme
theme = m.instruct(
    textwrap.dedent("""\
        Suggest a birthday party theme for a 10-year-old.
        Provide a theme name and a short explanation of why it suits this age group.
        All content must be age-appropriate.
    """)
)

# Subtask: list_activities
activities = m.instruct(
    textwrap.dedent("""\
        List at least 5 party activities suitable for children aged 8-12.
        Order the activities from most energetic to least energetic.
        Theme context: {{theme}}
        All content must be age-appropriate.
    """),
    user_variables={"theme": str(theme)},
)

# Subtask: catering_menu
menu = m.instruct(
    textwrap.dedent("""\
        Create a catering menu for a children's birthday party.
        Include a main dish, two sides, and a birthday cake option.
        All content must be age-appropriate.
    """)
)

# Subtask: invitation_message
invitation = m.instruct(
    textwrap.dedent("""\
        Write a birthday party invitation message addressed to the child's classmates.
        The message must not exceed 30 words.
        All content must be age-appropriate.
    """)
)

print("Theme:", str(theme))
print("Activities:", str(activities))
print("Menu:", str(menu))
print("Invitation:", str(invitation))
```

Each subtask is a separate `m.instruct()` call. Subtasks that depend on earlier
outputs receive them through `user_variables`. The file runs as-is:

```bash
python output/m_decomp_result.py
```

> **Note:** Generated output varies — LLM responses depend on model and
> temperature.

---

## Step 4: Refine the generated code

The generated script is a starting point. Common refinements:

### Add typed returns with `@generative`

Replace an `instruct()` call with a `@generative` function to get typed output
and IDE support:

```python
from typing import Literal
import mellea
from mellea import generative, start_session

@generative
def suggest_theme(age: int) -> str:
    """Suggest a birthday party theme for a child of the given age.
    Return a theme name followed by a one-sentence explanation."""

@generative
def list_activities(theme: str, age_min: int, age_max: int) -> list[str]:
    """List at least 5 party activities suitable for children aged age_min to age_max,
    ordered from most energetic to least energetic. All activities must be age-appropriate."""

m = start_session()
theme = suggest_theme(m, age=10)
activities = list_activities(m, theme=str(theme), age_min=8, age_max=12)

print(str(theme))
for activity in activities:
    print("-", activity)
# Output will vary — LLM responses depend on model and temperature.
```

### Add requirements to a subtask

Attach plain-English requirements to enforce constraints that `m decompose` left
as prose:

```python
import textwrap
import mellea
from mellea.stdlib.requirements import req, simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy

m = mellea.start_session()

invitation = m.instruct(
    textwrap.dedent("""\
        Write a birthday party invitation addressed to a 10-year-old's classmates.
        The message must not exceed 30 words. All content must be age-appropriate.
    """),
    requirements=[
        req(
            "Must not exceed 30 words.",
            validation_fn=simple_validate(
                lambda x: (
                    len(x.split()) <= 30,
                    f"Invitation is {len(x.split())} words; must be 30 or fewer.",
                )
            ),
        ),
        req("Must be addressed to classmates, not parents."),
    ],
    strategy=RejectionSamplingStrategy(loop_budget=4),
)
print(str(invitation))
# Output will vary — LLM responses depend on model and temperature.
```

---

## Step 5: Use --input-var for dynamic variables

When your prompt refers to values that change at runtime (a customer name, a
product ID, a date), declare them with `--input-var`. Variable names must be
valid Python identifiers, uppercase, and contain only alphanumeric characters
and underscores:

```bash
m decompose run \
  --prompt-file party_plan.txt \
  --out-dir ./output/ \
  --input-var CHILD_NAME \
  --input-var PARTY_DATE
```

The generated script will include placeholder references to `CHILD_NAME` and
`PARTY_DATE` as `user_variables`, ready for you to wire up at call time.

> **Warning:** `--input-var` names must be uppercase Python identifiers
> (e.g. `CHILD_NAME`, not `child-name` or `childName`). The command rejects
> names that contain hyphens, start with a digit, or use mixed case.

---

## Step 6: Choose the right model for decomposition

The decomposition quality depends heavily on the model. The default,
`mistral-small3.2:latest`, handles most prompts well. For more complex prompts
with many interdependent constraints, a larger model produces clearer subtask
boundaries:

```bash
m decompose run \
  --prompt-file party_plan.txt \
  --out-dir ./output/ \
  --model-id mistral-large:latest
```

To use an OpenAI-compatible endpoint:

```bash
m decompose run \
  --prompt-file party_plan.txt \
  --out-dir ./output/ \
  --backend openai \
  --model-id gpt-4o-mini \
  --backend-endpoint https://api.openai.com/v1 \
  --backend-api-key "$OPENAI_API_KEY"
```

> **Tip:** Run `m decompose run --help` to see the current defaults and all
> available flags.

---

## What the output JSON contains

The `.json` file gives you the full structured decomposition if you want to
process it programmatically:

```json
{
  "subtask_list": ["suggest_theme", "list_activities", "catering_menu", "invitation_message"],
  "identified_constraints": [
    {"constraint": "All content must be age-appropriate", "validation_strategy": "llm"},
    {"constraint": "Invitation must not exceed 30 words", "validation_strategy": "code"},
    {"constraint": "Activity list must be ordered from most energetic to least energetic", "validation_strategy": "llm"},
    {"constraint": "At least 5 activities", "validation_strategy": "code"},
    {"constraint": "Menu must include a main dish, two sides, and a birthday cake option", "validation_strategy": "llm"}
  ],
  "subtasks": [
    {
      "subtask": "Suggest a birthday party theme for a 10-year-old",
      "tag": "suggest_theme",
      "depends_on": [],
      "prompt_template": "Suggest a birthday party theme for a 10-year-old...",
      "input_vars_required": [],
      "constraints": [
        {"constraint": "All content must be age-appropriate", "validation_strategy": "llm"}
      ]
    }
  ]
}
```

Each subtask entry includes `depends_on` (a list of `tag` values), a ready-to-use
`prompt_template`, and the `constraints` that apply to it. Each constraint carries
a `validation_strategy` — `"code"` for deterministic checks (word count, length)
and `"llm"` for quality checks that require LLM-as-a-judge evaluation.

---

## Next steps

- [Generative Functions](../concepts/generative-functions) — add `@generative`,
  typed returns, and context steering to the generated pipeline
- [Enforce Structured Output](../how-to/enforce-structured-output) — constrain
  subtask outputs to Pydantic models or `Literal` values
- [CLI Reference](../reference/cli) — complete flag and option reference for
  all `m` subcommands
