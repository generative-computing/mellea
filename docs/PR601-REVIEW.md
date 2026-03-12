# PR #601 Review Comments — Working Tracker

Reviewers: **serjikibm**, **psschwei**, **HendrikStrobelt**

Status key: `[ ]` = open, `[x]` = done, `[~]` = won't fix / deferred, `[?]` = needs discussion

---

## Structural / High-level (psschwei)

- [ ] **H1 — Landing page duplication** (`index.mdx`)
  Docs landing page duplicates the separate marketing landing-page repo.
  Suggestion: open docs at installation or a thin index with section links.

- [ ] **H2 — Too much documentation / consolidation**
  - Merge guide + how-tos into one section
  - Fold evals & obs into how-to
  - Combine requirements + IVR concepts into one page
  - Merge glossary + troubleshooting into a "Reference" section
  - Deduplicate repeated code blocks (e.g. email requirements example)

- [ ] **H3 — Quickstart needs focus**
  Three examples is too many; consolidate to one with "wow factor".
  The "what's next" section at line 107 feels out of place — link out instead.
  Meta question: "what do we want folks to take away?"

- [ ] **H4 — Duplicate code blocks**
  e.g. email requirements appears in multiple places — consolidate.

---

## Broken Links (serjikibm) — 404s

- [ ] **L1** — `docs.json:327` — CONTRIBUTING link broken.
  Should be `https://github.com/generative-computing/mellea/blob/main/CONTRIBUTING.md`

- [ ] **L2** — `getting-started/quickstart.md:27` — link 404

- [ ] **L3** — `tutorials/01-your-first-generative-program.md:347` — example link 404

- [ ] **L4** — `tutorials/03-using-generative-slots.md:120` — example link 404

- [ ] **L5** — `tutorials/03-using-generative-slots.md:236` — example link 404

- [ ] **L6** — `tutorials/05-mifying-legacy-code.md:67` — link 404

- [ ] **L7** — `guide/m-decompose.md` (last serjikibm review) — link 404

---

## Installation / Shell Quoting (serjikibm + psschwei)

- [ ] **I1** — `installation.md:7` — Python version may need updating on next bump
  (Minor — note for future)

- [ ] **I2** — `installation.md:15` — Missing prerequisites: explain user needs
  uv-based venv and `uv init` before `uv add` will work.

- [ ] **I3** — `installation.md:26` — Inconsistent: offers `uv add` then switches
  to `pip`. **psschwei: default to uv only.**

- [ ] **I4** — `installation.md:26,36` — **zsh quoting** — `pip install mellea[litellm]`
  fails in zsh; must be `pip install "mellea[litellm]"`. Same for all `[extras]` installs.

- [ ] **I5** — `guide/backends-and-configuration.md` — Same zsh double-quote issue.

- [ ] **I6** — `guide/backends-and-configuration.md` — WatsonX env vars not documented.

---

## Missing Imports in Code Snippets (serjikibm)

- [ ] **M1** — `tutorials/03-using-generative-slots.md:61`
  Missing `from mellea import generative`

- [ ] **M2** — `tutorials/03-using-generative-slots.md:90`
  Not self-contained; needs note that it's a fragment, or add imports + class defs.

- [ ] **M3** — `tutorials/05-mifying-legacy-code.md:74,97,125`
  All three snippets missing `import mellea` and
  `from mellea.stdlib.components.mify import mify`

- [ ] **M4** — `tutorials/04-making-agents-reliable.md:292`
  Missing dependency `llguidance` — not installed by default.
  Needs `pip install llguidance` note.

---

## Code Snippet Runtime Errors (serjikibm)

These may be doc-only fixes or may indicate real API changes.

- [ ] **E1** — `tutorials/04-making-agents-reliable.md:201`
  Guardian check output confusing: deprecation warnings + "Guardian returned
  empty result" + false-positive safety failures. Is this expected?

- [ ] **E2** — `tutorials/04-making-agents-reliable.md:406`
  `MelleaTool.from_callable` crash:
  `AttributeError: 'MelleaTool' object has no attribute '__name__'`
  Likely passing a MelleaTool where a callable is expected.

- [ ] **E3** — `guide/tools-and-agents.md`
  Missing `ddgs` package for DuckDuckGo search example.
  Needs `uv pip install -U ddgs` note.

- [ ] **E4** — `guide/tools-and-agents.md`
  `AttributeError: 'ModelOutputThunk' object has no attribute 'body'`

- [ ] **E5** — `concepts/architecture-vs-agents.md`
  smolagents example: needs `pip install smolagents` note;
  gives incomplete response + serialization warning.

- [ ] **E6** — `concepts/architecture-vs-agents.md`
  LangChain `StructuredTool` import fails even after `pip install langchain`.
  Import path may have changed.

- [ ] **E7** — `concepts/mobjects-and-mify.md`
  Needs `pip install docling` note.
  Also: `ModuleNotFoundError: No module named 'mellea.stdlib.docs'`

- [ ] **E8** — `guide/act-and-aact.md`
  `NotImplementedError: parts isn't implemented by default` from
  `mellea/stdlib/components/docs/document.py`

- [ ] **E9** — `guide/m-decompose.md`
  CLI `m decompose`: output dir must pre-exist; pulls 15.2 GB model without
  warning; no cleanup/storage guidance.

---

## Content / Wording

- [ ] **C1** — `index.mdx:8` — Suggest alternative intro wording:
  "Mellea helps you manage the unreliable part…"

- [ ] **C2** — `index.mdx:37` — Cards-per-row inconsistent (2 then 3+).
  Lean towards uniform 2-per-row for readability.

- [ ] **C3** — `concepts/generative-functions.md` — Title casing:
  "functions" → "Functions" to match the how-to section heading.

- [ ] **C4** — `concepts/requirements-system.md` — Blog list link will become
  unhelpful as list grows. Link to specific post instead.

- [ ] **C5** — `concepts/instruct-validate-repair.md:182` — Explain dict/json
  key structure for context docs (is `doc0`/`doc1` mandatory or arbitrary?).

- [ ] **C6** — `tutorials/01-your-first-generative-program.md:38` — Include
  sample output, not just "output will vary".

- [ ] **C7** — `tutorials/01-your-first-generative-program.md:207` — Generative
  slots section duplicates tutorial 03. Remove from tutorial 01?

- [ ] **C8** — `tutorials/02-streaming-and-async.md:142` — Visual representation
  of streaming would help.

- [ ] **C9** — `tutorials/02-streaming-and-async.md:232` — Text says `await`
  suppresses deprecation warning, but it still appears. Fix text or example.

- [ ] **C10** — `guide/backends-and-configuration.md` — Expand LiteLLM section:
  self-hosted usage, `base_url`, how it differs from OpenAI backend type.

- [ ] **C11** — `guide/m-decompose.md` — Mixing programming-model concepts
  with CLI usage is confusing. Consider a dedicated CLI section.

---

## Misc

- [ ] **X1** — HendrikStrobelt: `.pre-commit-config.yaml` — markdownlint hook
  speed concern. "How fast is this? Might drag with many doc files."

- [ ] **X2** — psschwei: Quickstart identity question — "what do we want
  folks to take away?" Needs a single compelling example.

---

## Triage

### Fix now (mechanical — no design discussion needed)

- L1–L7: broken links
- I4, I5: zsh quoting
- M1–M4: missing imports
- C3: title capitalisation
- C6: add sample output
- E3: add `ddgs` install note

### Needs code investigation (may be bugs vs doc issues)

- E1: Guardian deprecation — is this expected output?
- E2: `MelleaTool.from_callable` crash
- E4: `ModelOutputThunk.body` AttributeError
- E6: LangChain `StructuredTool` import path
- E7: `mellea.stdlib.docs` missing module
- E8: `parts` NotImplementedError

### Needs discussion / design decisions

- H1–H4: structural reorganisation, landing page, quickstart
- I2, I3: uv-only install strategy
- C1, C2, C5, C7–C11: wording / content decisions
- E5, E9: third-party dependency warnings and large downloads
- X1: pre-commit hook performance
- X2: quickstart vision
