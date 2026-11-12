# Groundedness measurement set (#1721)

A labelled set and a harness for deciding #1669: replace `GroundednessRequirement`'s
two judgment prompts with the `hallucination_detection` adapter, or keep the prompts.

- `labelled_set.jsonl`: the items (fixtures; re-run them after any adapter bump).
- `harness.py`: runs an arm over the set, then compares two arms with the decision rule.
- `test_groundedness_eval.py`: unit tests for the set and the harness. No model needed.

## The set

128 items, all in English. Every fact is invented (a town, an e-bike, a clinic, a
river, a library, a railway, a recipe, a space probe, a leave policy, a museum, and
weather-station readings). Neither arm can then pass a claim from the base model's own
world knowledge. The set is written from scratch, so it has no third-party licence terms.

| Category | Items | Gold: strict / lenient | What it covers |
|---|---:|---|---|
| `grounded` | 30 | pass / pass | paraphrases of one document |
| `grounded_synthesis` | 20 | pass / pass | facts combined across documents, or simple arithmetic over them, with no quote |
| `partial` | 20 | fail / pass | a supported claim plus a detail the documents do not mention |
| `invented` | 30 | fail / fail | contradictions, and a supported sentence followed by an invented one |
| `non_factual` | 10 | pass / pass | greetings, thanks, clarifying questions, a refusal: no citation needed |
| `mixed_non_factual` | 10 | pass / pass | connecting text around supported claims |
| `long` | 8 | 4 pass, 4 fail | 20, 40, 60 and 80 sentences. In each failing one, the wrong sentence sits three-quarters of the way through, where truncated adapter output would lose it |

Each item has the response split into sentences, and each sentence has a gold label
(`faithful`, `partial`, `unfaithful`, `NA`, the adapter's own label set). The split is
exactly NLTK Punkt's, the splitter the adapter input uses, and a unit test checks this
for every item. The gold verdicts follow from the labels: strict passes only with no
`partial` and no `unfaithful`; lenient passes with no `unfaithful`.

Strict means `allow_partial_support=False`. Lenient lets `partial` items through, so
they are scored as "pass" in lenient mode. They are reported separately from invented
claims in strict mode.

Sizes: 74 gold-pass items in strict mode, 94 in lenient mode, and 34 items with an
invented claim (30 `invented` + 4 `long`).

## Arms

- **`prompts`** (arm 1): `GroundednessRequirement.validate` as it ships. Run it after
  #1718, so an errored result (`result.error` set) counts as a hard failure, not a verdict.
  Before #1718 such failures come back as ordinary "not grounded" verdicts and would
  inflate arm 1's false fails.
- **`adapter`** (arm 2): `find_citations` for the citation output, then the verdict from
  `flag_hallucinated_content` alone (`adapter_verdict` in `harness.py`):
  - any `unfaithful` sentence fails the response;
  - `partial` fails unless `allow_partial_support`;
  - `NA` is ignored;
  - a sentence with no record, an unknown label, or offsets outside the response is
    errored. A `ValueError` or `AdapterSchemaMismatchError` from either adapter call (bad
    JSON, truncated output, a contract error) is also errored. An errored validation
    detected nothing, so it is scored as wrong in both pools: a false fail on a gold-pass
    item and a miss on an invented-claim item. It also counts as a hard failure.

Run each arm in both modes. Arm 2 runs on both binding shapes (`--binding` labels the
results; `--backend-factory module:function` supplies a non-default backend such as
Granite Switch).

Any other exception stops the run: running out of memory, a failed adapter download or a
bad `--backend-factory` says nothing about the adapter, so it must not count against the
hard-failure gate.

## What is measured (per arm, mode and binding)

- False fails: gold-pass items that did not pass. This decides the swap.
- Misses: items with an invented claim that did not fail (errored counts as a miss).
- Strict mode only: partial-only items that passed.
- Label use (arm 2): gold sentence label against adapter label, so it shows whether
  `partial` and `NA` are used where they apply.
- Hard failures over all repeats, in total and by `error_type`. `ValueError` covers bad
  JSON and schema errors, but also input checks in the intrinsics, so the breakdown is
  there to audit the 2% figure.
- Stability: items whose verdict changes across repeats (`--repeats 3`, temperature 0).
- Latency (median, p90) and tokens per validation (errored validations are left out of
  the token means, since they stop partway). Tokens are summed from
  `generation.usage` through a scoped plugin. A row's totals stay empty unless every
  generation call in that validation reported usage, and the row keeps both counts. This
  depends on the call path, not only the backend: on `LocalHFBackend` adapter calls report
  no usage, so arm 2 has no totals there and neither does arm 1, whose `find_citations`
  call is an adapter call. Token costs on that binding need another source.
- Provenance, on every row: `--model`, `--repeats`, and the name, revision and binding
  type of each adapter that ran. Revision pinning covers the `local-file` binding only:
  the embedded (Granite Switch) path reports `revision: null`, so on that binding the
  mixed-revision check checks nothing and `--model` is the only provenance. Set `--model`
  to the served model id or Switch checkpoint when using `--backend-factory`.

Tokens and adapter provenance need the `hooks` extra. Without it the run warns and leaves
those fields empty, and `compare` refuses an adapter file without adapter provenance.

The first repeat is the primary verdict for every comparison. The other repeats only
feed the stability count.

## Running (GPU)

```bash
uv run python -m test.stdlib.requirements.groundedness_eval.harness run --arm prompts --out results/prompts_strict.jsonl
uv run python -m test.stdlib.requirements.groundedness_eval.harness run --arm prompts --allow-partial-support --out results/prompts_lenient.jsonl
uv run python -m test.stdlib.requirements.groundedness_eval.harness run --arm adapter --out results/adapter_strict.jsonl
uv run python -m test.stdlib.requirements.groundedness_eval.harness run --arm adapter --allow-partial-support --out results/adapter_lenient.jsonl
uv run python -m test.stdlib.requirements.groundedness_eval.harness compare results/prompts_strict.jsonl results/adapter_strict.jsonl --margin 0.05 --max-hard-failure 0.02 --rule margin
```

Use one `--out` file per arm, mode, binding and model. `compare` refuses a file that mixes
them, has items missing, stopped partway through or between repeats, or holds the wrong
arm, and it refuses two files with different models, `--repeats` or adapter revisions.
Run it once per binding. Resuming with a higher `--repeats` extends a run.

Runs append to `--out` and skip finished rows, so an interrupted run resumes. A run that
stopped on an exception wrote no row for that item, so resuming retries it. Recorded hard
failures are not retried: retrying until an item succeeds would hide flaky adapter
failures from the 2% gate. After changing the adapter, write to a new file.

## Decision rule: proposal, to be agreed on #1721 before any run

| | Proposed |
|---|---|
| Set | the 128 items above |
| Primary verdict | repeat 0 of 3 |
| Margin | 5 percentage points, for false fails and for misses, in each mode |
| "No worse" | rule `margin`: arm 2's rate is at most arm 1's + 5 points |
| Hard failures | arm 2 under 2% of validations, on each binding |

**This is a margin rule, with no significance test behind it.** It allows arm 2 one more
miss of the 34 invented-claim items, three more false fails of 74 in strict mode, and four
more of 94 in lenient mode. Exact McNemar cannot add anything here: whenever the margin
passes on this set, p is above 0.05, one-sided or two-sided
(`test_mcnemar_cannot_fire_inside_the_margin`). The test could only fire with about 120
invented-claim items. The p-value is still reported, for information.

**Why `margin`, not `ci`.** The harness also computes a strict non-inferiority reading
(`--rule ci`: the upper end of the 95% paired bootstrap CI must be within the margin). On
this set that reading is too demanding. Suppose the two arms each miss 3 of the 34
invented-claim items, but different ones. Their miss rates are equal, yet the CI is wide
enough to call arm 2 worse (`test_margin_and_ci_readings_differ_when_arms_tie_with_disagreements`).
Both readings are always reported, so the choice stays visible.
