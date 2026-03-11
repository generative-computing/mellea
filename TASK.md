# Task: Fix Ollama exception propagation in `generate_from_raw`

## Context

This branch has partial fixes for issues discovered in #573 (test flakiness on local/M1).
Most test fixes are already done (see git diff). One code fix remains.

## Remaining work

### Fix exception propagation in `OllamaModelBackend.generate_from_raw` (`mellea/backends/ollama.py`)

Currently `asyncio.gather(..., return_exceptions=True)` silently converts any exception into
`ModelOutputThunk(value="")`, storing the error only in `generate_log.extra["error"]` — invisible
to callers. A temporary `FancyLogger.warning()` was added as a diagnostic measure but the
exception is still swallowed.

**Fix:** remove `return_exceptions=True` so exceptions propagate naturally, consistent with all
other backends. Then remove the `isinstance(response, BaseException)` handling block and the
`FancyLogger.warning()` that was added as a temporary measure.

This aligns with the approach taken in #580 (fail fast, don't swallow exceptions) and gives
consistent behaviour across all backends.

## Related issues / PRs

- #573 — test flakiness (this branch addresses most items)
- #577 — same root cause in streaming path
- #580 — fix for streaming path (`astream()` in `base.py`), now merged
- #432 — original bug report covering all backends; closes when #580 merges

## What's already done in this branch

- `researcher.py` — `slow` marker added
- `test_format` — `MAX_NEW_TOKENS` bumped to 1024 (ollama + openai_ollama)
- `test_generate_from_raw_with_format` — stale `xfail` removed, assertions validate all results
- `test_generate_from_raw` — 150s timeout added, tighter assertions, reduced context window (2048) **[caveat: see below]**
- `slow` marker description updated (">1 minute")
- Stale `xfail` removed from `test_litellm_watsonx.py`

## Caveats / open questions

### `CONTEXT_WINDOW: 2048` in `test_generate_from_raw` may itself cause failures

Added to reduce KV cache pressure (32K × 4 parallel = 128K slots → 2K × 4 = 8K). In
theory harmless for simple arithmetic prompts. However a 20-run soak after an extended
test session (machine under sustained load) showed 18/20 failures, with Ollama returning
**empty-body responses** (`response.response == ""`) rather than exceptions or hangs.
The `assert all(r.value for r in results)` assertion was catching these empty responses —
the FancyLogger.warning() did NOT fire (no `BaseException` in the gather results), confirming
the empty string was a legitimate Ollama response, not a caught exception.

This may be caused by the reduced context window interacting with Ollama's internal state
under load, or simply by machine exhaustion — a cold run after the machine rests is needed
to distinguish. The `CONTEXT_WINDOW: 2048` change should be treated as provisional until
this is resolved.

### aiohttp unclosed session reference

TASK.md cites "litellm bug #11657" for the unclosed aiohttp `ClientSession` at suite
teardown — confirm this issue number is correct before referencing it in a PR.

## Out of scope

- Unclosed aiohttp `ClientSession` at teardown — blocked on upstream litellm bug #11657
- `asyncio_default_fixture_loop_scope` pytest-asyncio warning — low priority
- The intermittent Ollama hang under load — still under investigation; fixing exception
  propagation will at least make failures visible rather than silent
