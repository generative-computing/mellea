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
  > **Looking to use this in code?** See [Generative Functions](../guide/generative-functions) for practical examples and API details.
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
- Model IDs current (e.g., `ibm-granite/granite-4.0-micro`).
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

**Linking rule:** Cross-link to the glossary on **first use only** of a term on each page — not every occurrence. Use anchor links, e.g. `[`MelleaSession`](../guide/glossary#melleasession)`.

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
> **Full example:** [`docs/examples/tutorial/simple_email.py`](../../examples/tutorial/simple_email.py)
```

Only link examples that are current and in CI.

---

## Missing content

If content is genuinely missing (no source, needs input from the team), open a GitHub issue and track it there. **Do not leave visible placeholders or "TODO" markers in published pages.**

---

## Page length

Target 300–600 lines. Split if >800. If a page is hard to read in one sitting without losing your place, split it.

---

## Navigation footer

Every page ends with a navigation footer:

```markdown
---


**See also:** [Related Page](./related), [Another Page](./another)
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

---

## Local preview

```bash
cd docs/docs
mint dev
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
- [ ] Navigation footer present (Next + See also).
- [ ] `docs.json` updated if new page added; old MDX page removed from nav if replaced.
- [ ] `index.mdx` landing page cards reviewed — add a card if the new page is a major entry point (key pattern, integration, or prominent how-to); keep total cards per section to ≤ 8.
- [ ] Previewed locally with `mint dev`.
- [ ] Non-deterministic LLM output noted.
- [ ] Backend-specific code blocks flagged with `> **Backend note:**`.
- [ ] No visible TODO placeholders — missing content tracked as GitHub issues.
- [ ] `# diataxis:` comment in frontmatter.
- [ ] If the page has a paired explanation/how-to counterpart, cross-link added near the top of both pages (see "Cross-linking paired pages").
