---
name: skill-author
description: >
  Draft, validate, and install new agent skills. Use when asked to create a new
  skill, automate a workflow, or add a capability. Produces cross-compatible
  SKILL.md files that work in both Claude Code and IBM Bob.
argument-hint: "[skill-name]"
compatibility: "Claude Code, IBM Bob"
metadata:
  version: "2026-03-25"
  capabilities: [bash, read_file, write_file]
---

# Skill Authoring Meta-Skill

Create new agent skills that work across Claude Code (CLI/IDE) and IBM Bob.

## Skill Location

Skills live under `.agents/skills/<name>/SKILL.md`.

Discovery configuration varies by tool:
- **Claude Code:** Add `"skillLocations": [".agents/skills"]` to `.claude/settings.json`.
  Without this, Claude Code looks in `.claude/skills/` by default.
- **IBM Bob:** Discovers `.agents/skills/` natively per agentskills.io convention.

Both tools read the same `SKILL.md` format. Use the frontmatter schema below
to maximise compatibility.

## Workflow

1. **Name the skill** — kebab-case, max 64 chars (e.g. `api-tester`, `audit-markers`).

2. **Scaffold the directory:**
   ```
   .agents/skills/<name>/
   ├── SKILL.md          # Required — frontmatter + instructions
   ├── scripts/          # Optional — helper scripts
   └── templates/        # Optional — output templates
   ```

3. **Write SKILL.md** — YAML frontmatter + markdown body (see schema below).

4. **Dry-run review** — mentally execute the skill against a realistic scenario
   before finalising. Walk through the procedure on a concrete example (a real
   file in the repo, not a hypothetical) and check for:
   - **Scaling gaps:** Does the procedure work for 1 file AND 100 files? If the
     skill accepts a directory or glob, it needs a triage strategy (e.g., "grep
     first to find candidates, then deep-read only files with issues") — not
     just "read every file fully."
   - **Boundary ambiguity:** If the skill defines categories or classifications,
     test the boundaries between adjacent categories with a real example. The
     edges are where agents will disagree or ask the user. Sharpen definitions
     until two agents reading the same test would classify it the same way.
   - **Stale references:** If the skill describes project state ("this hook needs
     to be added", "this marker is not yet registered"), verify those statements
     are still true. Embed checks ("read conftest.py to confirm") rather than
     assertions that rot.
   - **Output format at scale:** Run the report template mentally against the
     largest expected input. A per-function report for 5 files is fine; for 165
     files it's unusable. Design output for the largest scope — summary table
     first, per-item detail only where issues exist.
   - **Format coverage:** If the skill operates on multiple input formats (e.g.,
     `pytestmark` lists AND `# pytest:` comments), verify each format is
     explicitly addressed in the procedure. Implicit coverage causes agents to
     skip or guess.
   - **Rigid rules:** If you wrote "always X" or "never Y", find the edge case
     where the rule is wrong. Add the escape hatch. E.g., "per-function only"
     should say "module-level is acceptable when every function qualifies."

5. **Validate:**
   - Check the skill is discoverable: list files in `.agents/skills/`.
   - Confirm no frontmatter warnings from the IDE.
   - Verify the skill does not conflict with existing skills or `AGENTS.md`.

## SKILL.md Frontmatter Schema

Use only fields from the **cross-compatible** set to avoid IDE warnings.

### Cross-compatible fields (use these)

| Field | Type | Purpose |
|-------|------|---------|
| `name` | string | Kebab-case identifier. Becomes the `/slash-command`. Max 64 chars. |
| `description` | string | What the skill does and when to trigger it. Be specific — agents use this to decide whether to invoke the skill automatically. |
| `argument-hint` | string | Autocomplete hint. E.g. `"[file] [--dry-run]"`, `"[issue-number]"`. |
| `compatibility` | string | Which tools support this skill. E.g. `"Claude Code, IBM Bob"`. |
| `disable-model-invocation` | boolean | `true` = manual `/name` only, no auto-invocation. |
| `user-invocable` | boolean | `false` = hidden from `/` menu. Use for background knowledge skills. |
| `license` | string | SPDX identifier if publishing. E.g. `"Apache-2.0"`. |
| `metadata` | object | Free-form key-value pairs for tool-specific or custom fields. |

### Tool-specific fields (put under `metadata`)

These are useful but not universally supported — nest them under `metadata`:

```yaml
metadata:
  version: "2026-03-25"
  capabilities: [bash, read_file, write_file]   # Bob/agentskills.io
```

Claude Code's `allowed-tools` and `context`/`agent` fields are recognised by
Claude Code but may trigger warnings in Bob's validator. If needed, add them
to `metadata` or accept the warnings.

### Example frontmatter

```yaml
---
name: my-skill
description: >
  Does X when Y. Use when asked to Z.
argument-hint: "[target] [--flag]"
compatibility: "Claude Code, IBM Bob"
metadata:
  version: "2026-03-25"
  capabilities: [bash, read_file, write_file]
---
```

## SKILL.md Body Structure

After frontmatter, write clear markdown instructions the agent follows:

1. **Context section** — what the skill operates on, key reference files.
2. **Procedure** — numbered steps the agent follows. Be explicit about decisions and edge cases.
3. **Rules / constraints** — hard rules the agent must not break.
4. **Output format** — what the agent should produce (report, edits, summary).

### Guidelines

- **Be specific.** Vague instructions produce inconsistent results across models.
  "Check if markers are correct" is worse than "Compare the test's assertions
  to the qualitative decision rule in section 3."
- **Reference project files.** Point to docs, configs, and examples by relative
  path so the agent can read them. E.g. "See `test/MARKERS_GUIDE.md` for the
  full marker taxonomy."
- **Declare scope boundaries.** State what the skill does NOT do. E.g. "This
  skill does not modify conftest.py — flag infrastructure issues as notes."
- **Use `$ARGUMENTS`** for user input. `$ARGUMENTS` is the full argument string;
  `$1`, `$2` etc. are positional.
- **Keep SKILL.md under 500 lines.** Use supporting files for large reference
  material (link to them from the body).
- **Portability:** use relative paths from the repo root, never absolute paths.
- **Formatting:** use YYYY-MM-DD for dates, 24-hour clock for times, metric units.
- **Design for variable scope.** If the skill can operate on a single file or an
  entire directory, provide a triage strategy for the large case. Agents given
  "audit everything" with no prioritisation will either read every file (slow)
  or skip files (incomplete).
- **Sharpen category boundaries.** When defining classifications, the boundary
  between adjacent categories causes the most disagreement. Add a "key
  distinction from X" sentence for each pair of adjacent tiers.
- **Avoid temporal assertions.** Don't write "this conftest hook needs to be
  added" — write "check whether conftest.py already has the hook." State that
  goes stale silently is worse than no guidance at all.
- **Qualify absolutes.** "Always X" and "never Y" rules need escape hatches for
  the common exception. E.g., "per-function only — unless every function in the
  file qualifies, in which case module-level is acceptable."
