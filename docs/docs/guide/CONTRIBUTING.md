---
title: "Contributing to the Mellea docs"
description: "Writing conventions, review process, and PR checklist for Mellea guide pages."
# diataxis: reference
---

# Contributing to the Mellea docs

This file is the authoritative writing guide for `docs/docs/guide/`. It is linked from the root `CONTRIBUTING.md` and is also accessible on the published docs site.

---

## Core principle: progressive disclosure

The nav IS the progressive learning path:

> Introduction → Quick Start → Core Concepts → Extending Mellea → Internals

Each section assumes the previous. Within a page: working code first, then explain it. Common case before edge cases. Mark advanced content with `> **Advanced:**`. Conceptual depth belongs in dedicated pages, not scattered through how-to pages.

---

## Audience

Python developers who know Python, likely know Pydantic, understand LLM basics. Some readers are true AI research experts — never condescend, never over-explain Python/Pydantic basics.

- Introduce Mellea-specific concepts on first use; link out for deeper context.
- Never use "simply", "just", "easy", "obviously", "straightforward".
- Each page should be useful at a shallow read AND reward deeper reading.

---

## Language

**US English** throughout, including code comments: "behavior", "color", "recognize", "initialize". Matches the Mellea source code.

---

## Frontmatter (required on every page)

```yaml
---
title: "Getting Started"
description: "Install Mellea and run your first generative program in minutes."
# diataxis: tutorial
---
```

`sidebarTitle` is optional — add only when `title` is too long for the nav sidebar.

The `# diataxis:` comment is for contributors; it is not rendered to readers.

### Diataxis classification

Add a `# diataxis:` comment in every page's frontmatter:

| Value | Use for |
| ----- | ------- |
| `tutorial` | Learning-oriented, follow-along (e.g., `getting-started`) |
| `how-to` | Task-oriented (e.g., `tools-and-agents`, `working-with-data`) |
| `reference` | Information-oriented (e.g., `glossary`, API docs) |
| `explanation` | Understanding-oriented (e.g., `generative-programming`, `internals`) |

### Cross-linking paired pages

Some features have two pages: an **explanation** page in `concepts/` (what it
is and why it works the way it does) and a **how-to** page in `guide/` or
`how-to/` (how to use it). Both are valid entry points — a reader may land on
either depending on how they searched.

When a feature has paired pages, add a brief cross-link near the top of each,
before the first H2, so readers can orient themselves quickly:

- On the **explanation** page:

  ```markdown
  > **Looking to use this in code?** See [Generative Functions](../how-to/generative-functions) for practical examples and API details.
  ```

- On the **how-to** page:

  ```markdown
  > **Concept overview:** [Generative functions](../concepts/generative-functions) explains the design and trade-offs.
  ```

Keep both cross-links to one sentence. Do not duplicate content between the
two pages — the explanation should cover *why*, the how-to should cover *how*.

---

## Headings

- No H1 — Mintlify renders the frontmatter `title` as the page heading automatically. Start body content with H2.
- H2 = major sections; H3 = subsections. Never skip heading levels.
- Sentence case: "Working with data", not "Working With Data".

---

## Code blocks

Every fenced block **must** have a language tag.

| Content | Tag |
| ------- | --- |
| Python | `python` |
| Shell / terminal | `bash` |
| JSON | `json` |
| YAML | `yaml` |
| Plain text output | `text` |
| Interactive console | `console` |

Rules:

- Always include all necessary imports — never assume they carry over from a prior block.
- Include type hints where they aid clarity; omit or simplify where they obscure.
- Show expected output as a `# comment` or `text` block where it helps the reader.
- Keep examples minimal but complete — no unexplained variables.
- Prefer real-world examples over abstract `foo`/`bar`.
- Inline `python` examples must be syntactically correct and runnable in the context established by the page's prerequisites block. They are not required to be self-contained standalone scripts.
- Fully standalone examples belong in `docs/examples/` where CI will test them. Link with `> **Full example:**`. Inline examples in guide pages are verified by human review at PR time.
- Keep inline examples to ~20–30 lines. If more is needed, move it to `docs/examples/`.

**Non-deterministic output:** When showing LLM-generated text, note variance:

```python
print(result.value)
# Output will vary — LLM responses depend on model and temperature.
```

Or a section-level callout if multiple blocks share the caveat:

```text
> **Note:** LLM output is non-deterministic. Your exact results will vary.
```

---

## Code and fragment consistency

All code — fenced blocks AND inline backtick references — must match current source:

- Import paths, class names, method names exact.
- Model IDs current (e.g., `ibm-granite/granite-4.1-3b`).
- Inline prose fragments consistent with adjacent code blocks.

If the source itself has inconsistencies, document as-is and note in the glossary.

---

## API keys and credentials

Always use placeholders: `api_key="sk-..."`, `api_key="your-api-key-here"`. Never anything that resembles a real key.

---

## Prerequisites

Procedural pages open with a prerequisites block before the first code example:

```markdown
**Prerequisites:** [Ollama](https://ollama.ai) installed and running, `pip install mellea` complete.
```

State only what is genuinely required for that specific page.

---

## Lists

- **Numbered** for sequential steps (order matters).
- **Bullets** for unordered items (features, options, caveats).

---

## Links

- Within guide: relative — `./tools-and-agents.md`
- API reference: from docs root — `../../api/mellea/stdlib/session`
- External: descriptive text — `[Ollama](https://ollama.ai)` — no bare URLs.

Verify before merge: relative links resolve, absolute URLs return HTTP 200.

---

## Glossary and terminology

`glossary.md` defines all Mellea-specific terms. Use canonical terms from the glossary; never invent synonyms. Add new terms to `glossary.md` as you write each page.

**Linking rule:** Cross-link to the glossary on **first use only** of a term on each page — not every occurrence. Use anchor links, e.g. `[`MelleaSession`](../reference/glossary#melleasession)`.

Terms that **must** be linked on first use wherever they appear in guide pages (getting-started, tutorials, concepts, how-to, integrations, advanced):

| Term | Anchor |
| ---- | ------ |
| `@generative` / generative function | `#generative` |
| `MelleaSession` / `start_session()` | `#melleasession` |
| `ModelOutputThunk` | `#modeloutputthunk` |
| `SamplingResult` | `#samplingresult` |
| `SimpleContext` / `ChatContext` | `#context` |
| `Component` | `#component` |
| `Backend` | `#backend` |
| `Requirement` / `req()` / `check()` | `#requirement` |
| IVR / Instruct–Validate–Repair | `#ivr-instruct-validate-repair` |
| Sampling strategy / `RejectionSamplingStrategy` etc. | `#sampling-strategy` |
| `ModelOption` | `#modeloption` |
| `MObject` / `@mify` | `#mobject` / `#mify--mify` |
| `aLoRA` | `#alora-activated-lora` |
| `ReAct` | `#react` |
| `RichDocument` | `#richdocument` |
| `LiteLLM` / `LiteLLMBackend` | `#litellm--litellmbackend` |
| `GuardianCheck` / `GuardianRisk` | `#guardiancheck` |
| `m decompose` | `#m-decompose` |

Linking within the **glossary page itself** is not required (the glossary is the definition source).

---

## Callouts

Three core types (plain markdown, no JSX):

```markdown
> **Note:** Worth knowing but not blocking.
> **Warning:** Will break or cause unexpected behavior.
> **Advanced:** Safe to skip on first read.
```

For other needs, handle inline:

- Deprecations: `> **Deprecated in vX.x:** Use Y instead.`
- Coming-soon content: `> **Coming soon:** Planned for a future release.`
- Backend-specific code: `> **Backend note:** This example requires [Backend]. Other backends may differ.`

Use **Backend note:** whenever a code block or behavior is specific to one provider (e.g., Ollama, OpenAI, Bedrock, WatsonX).

---

## Error output

Show what failure modes actually look like in a `text` block. If the exact message varies by backend or version, add a `> **Note:**`. If an example can't be produced now, track it as a GitHub issue — don't leave a placeholder in published docs.

---

## Full example pointers

Where a CI-tested example exists in `docs/examples/`, link it:

```text
> **Full example:** [`docs/examples/tutorial/simple_email.py`](https://github.com/generative-computing/mellea/blob/main/docs/examples/tutorial/simple_email.py)
```

Only link examples that are current and in CI.

### Keeping the examples catalogue up to date

Check that every example directory under `docs/examples/` has a corresponding
row in the catalogue table in `docs/docs/examples/index.md`. When adding a new
example directory, add a row with a short description of what the examples
demonstrate. This ensures users browsing the published docs can discover all
available examples, not only the ones linked from individual guide or concept
pages.

---

## Missing content

If content is genuinely missing (no source, needs input from the team), open a GitHub issue and track it there. **Do not leave visible placeholders or "TODO" markers in published pages.**

---

## Page length

Target 300–600 lines. Split if >800. If a page is hard to read in one sitting without losing your place, split it.

---

## Navigation footer

Mintlify renders previous/next page links automatically from the nav order in `docs.json` — do not add these manually. Add a `**See also:**` block at the end of each page for non-sequential cross-links:

```markdown
---


**See also:** [Glossary](./glossary), [Working with Data](./working-with-data)
```

---

## Voice and tone

- **Concise.** Cut every sentence that doesn't add meaning.
- Active voice, second person, present tense.
- Section intro: one sentence on what this section covers and why it matters.
- No padding: "In this section we will...", "As mentioned above...", "It is worth noting that...".

---

## Versioning

No version tags on individual features yet — incomplete tagging misleads readers. Tracked separately in issue #557.

---

## Deprecation

```text
> **Deprecated in v0.x:** `old_method()` is removed. Use `new_method()` instead.
```

---

## Docstrings (for code contributors)

Mellea uses **Google-style docstrings**. These feed the auto-generated API reference.

**Functions** — `Args:` and `Returns:` on the function docstring:

```python
def my_function(arg: str) -> bool:
    """One-line summary.

    Args:
        arg: Description of the argument.

    Returns:
        Description of the return value.

    Raises:
        ValueError: When and why this is raised.
    """
```

**Classes** — `Args:` on the *class* docstring only; `__init__` gets a single summary sentence.
The docs pipeline skips `__init__`, so `Args:` must live on the class to appear in the API reference:

```python
class MyComponent(Component[str]):
    """A component that does something useful.

    Args:
        name (str): Human-readable label for this component.
        max_tokens (int): Upper bound on generated tokens.
    """

    def __init__(self, name: str, max_tokens: int = 256) -> None:
        """Initialize MyComponent with a name and token budget."""
        self.name = name
        self.max_tokens = max_tokens
```

Add `Attributes:` only when a stored value differs in type or behaviour from the constructor
input (e.g. a `str` wrapped into a `CBlock`, or a class-level constant).
Pure-echo entries that repeat `Args:` verbatim should be omitted.

**`TypedDict` classes** are a special case — their fields are the entire public contract,
so when an `Attributes:` section is present it must exactly match the declared fields.
The CI audit will fail on phantom fields (documented but not declared) and undocumented
fields (declared but missing from `Attributes:`).

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for the full validation workflow.

### CI docstring checks reference

The `audit_coverage.py --quality` gate (run in CI on every PR) reports the
following check kinds. If your PR is blocked by this gate, find the check kind
in the table below, follow the fix instructions, and re-push.

> **Note on PR diff annotations:** GitHub Actions shows inline annotations
> directly on the changed lines in your PR diff. These are capped at **10 per
> check category** to ensure every category gets at least one visible marker.
> If there are more issues than the cap, a `"... and N more"` notice appears in
> the job log. The **complete list** of issues is always in the full job log
> (expand the "Docstring quality gate" step) and in the
> `docstring-quality-report` artifact (JSON) attached to the workflow run.

#### Missing or short docstrings

| Check kind | What it means | How to fix |
| ---------- | ------------- | ---------- |
| `missing` | No docstring present on the symbol | Add a Google-style one-line summary sentence |
| `short` | Docstring has fewer than 5 words | Expand the summary — describe what the function/class does and why |

#### Args, Returns, Yields, and Raises

| Check kind | What it means | How to fix |
| ---------- | ------------- | ---------- |
| `no_args` | Function has named parameters but the docstring has no `Args:` section | Add an `Args:` section listing each parameter name and a short description |
| `no_returns` | Function has a non-`None` return annotation but the docstring has no `Returns:` section | Add a `Returns:` section describing what is returned and when |
| `no_yields` | Function returns `Generator` / `Iterator` but the docstring has no `Yields:` section | Add a `Yields:` section — generator functions use `Yields:`, not `Returns:` |
| `no_raises` | Function source contains `raise` but the docstring has no `Raises:` section | Add a `Raises:` section listing each exception type and the condition that triggers it |
| `missing_param_type` | `Args:` section exists but one or more parameters have no Python type annotation — the type column is absent from the generated API docs | Add a type annotation to each listed parameter in the function signature (e.g. `def f(x: int)`). Only fires when `no_args` is already satisfied; `*args`/`**kwargs` are excluded. |
| `missing_return_type` | `Returns:` section is documented but the function has no return type annotation — the return type is absent from the generated API docs | Add a return annotation to the function signature (e.g. `-> str`). Only fires when `no_returns` is already satisfied. |
| `param_type_mismatch` | A parameter's `Args:` entry states an explicit type (e.g. `x (int): …`) that does not match the Python annotation in the function signature | Align the docstring type with the annotation, or vice versa. The check normalises common equivalents (`Optional[X]` ↔ `X \| None`, `List` ↔ `list`, union ordering) before comparing, so only genuine disagreements are flagged. Only fires when both the docstring and the signature have an explicit type. **Note:** Python's AST normalises string literals to single quotes, so `Literal["a", "b"]` in source is read as `Literal['a', 'b']` — use single quotes in docstrings to match. |
| `return_type_mismatch` | The `Returns:` section has a type prefix (e.g. `Returns: \n    str: …`) that does not match the Python return annotation | Align the docstring return type with the annotation, or vice versa. Same normalisation rules as `param_type_mismatch`. Only fires when both sides have an explicit type. |

#### Class docstrings (Option C)

| Check kind | What it means | How to fix |
| ---------- | ------------- | ---------- |
| `no_class_args` | Class `__init__` has typed parameters but the **class** docstring has no `Args:` section | Add `Args:` to the class docstring (not `__init__`) — see Option C convention above |
| `duplicate_init_args` | `Args:` appears in **both** the class docstring and the `__init__` docstring | Remove `Args:` from the `__init__` docstring; keep it on the class docstring only |
| `param_mismatch` | `Args:` section documents parameter names that do not exist in the actual signature | Remove or rename the phantom entries so they exactly match the real parameter names |

#### TypedDict classes

| Check kind | What it means | How to fix |
| ---------- | ------------- | ---------- |
| `typeddict_phantom` | `Attributes:` section documents field names not declared in the `TypedDict` | Remove the extra entries — every `Attributes:` entry must match a declared field |
| `typeddict_undocumented` | `TypedDict` has declared fields that are absent from the `Attributes:` section | Add the missing fields — every declared field must appear in `Attributes:` |

---

## CLI command docstrings

Typer command functions in `cli/` feed the auto-generated **CLI Reference** page
(`docs/docs/reference/cli.md`). The generator script
(`tooling/docs-autogen/generate_cli_reference.py`) extracts content from both
Typer metadata (`help=` strings on `typer.Option`/`typer.Argument`) and the
command function's docstring.

Follow this convention for CLI command functions:

```python
def my_command(
    path: str = typer.Argument(..., help="File or directory to process"),
    verbose: bool = typer.Option(False, help="Enable verbose output"),
):
    """One-line summary of what the command does.

    Extended description with more detail about the command's behaviour.

    Prerequisites:
        Mellea installed (``uv add mellea``). Any other requirements
        (running services, authentication, etc.).

    Output:
        Describe what the command produces — files written, services started,
        or side effects applied.

    Examples:
        m my-command path/to/input --flag value

    See Also:
        guide: how-to/some-guide-page
        guide: advanced/another-page
    """
```

**Rules:**

- **First line** — imperative summary; becomes the command description in the reference.
- **Body** — expanded description; rendered as a paragraph below the summary.
- **`Prerequisites:`** — what must be installed or running. Rendered as a callout.
- **`Output:`** — what the command produces (files, services, side effects). Rendered
  as an "Output" paragraph.
- **`Examples:`** — a minimal one-liner invocation showing the most common flags.
  Rendered as a fenced code block.
- **`See Also:`** — cross-links to guide pages. Each line is `guide: <relative-doc-path>`
  (no `.md` extension). Rendered as "See also" links.
- **`help=` strings** on `typer.Option()` / `typer.Argument()` become the flag
  descriptions in the options table. Every option **must** have a `help=` string.
- Regenerate after changes: `uv run poe clidocs`

### CI enforcement

The build pipeline runs `generate_cli_reference.py --strict` which fails if any
command is missing a summary, `Prerequisites:`, `Output:` section, or has options
without `help=` text. The `docs-publish` workflow also runs the docs-autogen unit
tests which verify that all expected commands appear in the generated output.

### Cross-linking convention

Guide pages that document CLI commands should include a link to the CLI Reference
in their **See also** footer:

```markdown
**See also:** [Other Page](../path/to-page) | [CLI Reference](../reference/cli)
```

---

## Local preview

```bash
cd docs/docs
mintlify dev
# Site available at http://localhost:3000
```

---

## Linting

All guide pages must pass `markdownlint` with zero warnings **per page before moving on**. Config: `docs/docs/guide/.markdownlint.json`.

```bash
markdownlint docs/docs/guide/your-page.md
```

---

## Images

- Store in `docs/docs/guide/images/`, relative paths, always include alt text.
- Prefer text or code over images where possible.

---

## Review process

1. Author (Nigel or contributor) — self-review against this checklist.
2. Hendrik — technical accuracy review.
3. PR — broader team review before merge.

---

## PR checklist

- [ ] All code blocks have language tags.
- [ ] All code and inline fragments verified against current Mellea source.
- [ ] No real API keys or credentials.
- [ ] All relative links resolve; external links checked.
- [ ] US English throughout, including code comments.
- [ ] `markdownlint` passes with zero warnings.
- [ ] New glossary terms added to `glossary.md`.
- [ ] Mellea-specific terms linked to `glossary.md` on first use (see "Glossary and terminology" section).
- [ ] `**See also:**` footer present with relevant cross-links (Mintlify generates prev/next automatically).
- [ ] `docs.json` updated if new page added; old MDX page removed from nav if replaced.
- [ ] `index.mdx` landing page cards reviewed — add a card if the new page is a major entry point (key pattern, integration, or prominent how-to); keep total cards per section to ≤ 8.
- [ ] Previewed locally with `mintlify dev`.
- [ ] Non-deterministic LLM output noted.
- [ ] Backend-specific code blocks flagged with `> **Backend note:**`.
- [ ] No visible TODO placeholders — missing content tracked as GitHub issues.
- [ ] `# diataxis:` comment in frontmatter.
- [ ] If the page has a paired explanation/how-to counterpart, cross-link added near the top of both pages (see "Cross-linking paired pages").
