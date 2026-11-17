# Groundedness measurement set (#1721)

A labelled set and a harness for deciding #1669: replace `GroundednessRequirement`'s
two judgment prompts with the `hallucination_detection` adapter, or keep the prompts.

- `labelled_set.jsonl`: the items (fixtures; re-run them after any adapter bump).
- `harness.py`: runs an arm over the set, then applies the decision rule to both arms.
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
  inflate arm 1's false fails. In production an errored result fails closed and the claim
  is blocked, so an errored arm 1 row is scored as a false fail on a gold-pass item and a
  catch on a gold-fail one. This matters on the long items: arm 1 caps each judgment call
  at 500 tokens (`max_new_tokens`), which holds only the first 30 or so spans of the
  necessity JSON, and the wrong sentence in each failing long item comes after that.
- **`adapter`** (arm 2): `find_citations` for the citation output, then the verdict from
  `flag_hallucinated_content` alone (`adapter_verdict` in `harness.py`):
  - any `unfaithful` sentence fails the response;
  - `partial` fails unless `allow_partial_support`;
  - `NA` is ignored;
  - a sentence with no record, an unknown label, or offsets outside the response is
    errored. A `ValueError` or `AdapterSchemaMismatchError` from either adapter call (bad
    JSON, truncated output, a contract error) is also errored. An errored arm 2 validation
    is what is being measured and detected nothing, so it is scored as wrong in every
    pool: a false fail on a gold-pass item and a miss on a gold-fail one. It also counts
    toward the hard-failure gate.

Run each arm in both modes. Arm 2 runs on both binding shapes (`--binding` labels the
results; `--backend-factory module:function` supplies a non-default backend such as
Granite Switch).

Any other exception stops the run: running out of memory, a failed adapter download or a
bad `--backend-factory` says nothing about the adapter, so it must not count against the
hard-failure gate.

## What is measured (per arm, mode and binding)

- False fails: gold-pass items that did not pass. This decides the swap.
- Misses: items with an invented claim that did not fail (an errored row is a miss for arm
  2 and a catch for arm 1).
- Strict mode only: partial-only items that passed (errored rows as for misses). This is
  gated too: the 20 partial items are gold-fail in strict mode, so neither of the other
  two errors covers them.
- Label use (arm 2): gold sentence label against adapter label, so it shows whether
  `partial` and `NA` are used where they apply.
- Hard failures over all repeats, in total and by `error_type`. `ValueError` covers bad
  JSON and schema errors, but also input checks in the intrinsics, so the breakdown is
  there to audit the hard-failure gate. The gate counts items with an errored repeat.
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
uv run python -m test.stdlib.requirements.groundedness_eval.harness compare --strict results/prompts_strict.jsonl results/adapter_strict.jsonl --lenient results/prompts_lenient.jsonl results/adapter_lenient.jsonl
```

Use one `--out` file per arm, mode, binding and model. `compare` refuses a file that mixes
them, has items missing, stopped partway through or between repeats, or holds the wrong
arm, and it refuses two files with different models, `--repeats` or adapter revisions,
or strict and lenient runs that differ in model, binding or adapter revisions. It takes
one binding's four files and prints the verdict; run it once per binding. Resuming with a higher `--repeats` extends a run.

Runs append to `--out` and skip finished rows, so an interrupted run resumes. A run that
stopped on an exception wrote no row for that item, so resuming retries it. Recorded hard
failures are not retried: retrying until an item succeeds would hide flaky adapter
failures from the hard-failure gate. After changing the adapter, write to a new file.

## Decision rule (#1721), fixed before any run

Every check counts items on the primary repeat (repeat 0). A gated check passes when arm 2
gets at most its allowance more items wrong than arm 1.

| | Rule |
|---|---|
| Set | the 128 items above |
| Gated, strict | false fails +3 of 74, misses +1 of 34, partial-only passes +1 of 20 |
| Gated, lenient | misses +1 of 34; lenient false fails (94 items) are reported, not gated |
| Errored rows | arm 2 wrong everywhere; arm 1 a false fail on gold-pass items, a catch on gold-fail |
| Binding that decides | `local-file`, both arms on base model `ibm-granite/granite-4.0-micro`; `citations` and `hallucination_detection` LoRAs from `ibm-granite/granitelib-rag-r1.0` at `2f0b2c79` (the catalog pin; both publish `granite-4.0-micro` there). Every row records the revisions and `compare` refuses a mix. Granite Switch is reported |
| Hard failures | arm 2, at most 2 items with an errored repeat, over both modes, on each binding |
| Primary verdict | repeat 0 of 3; repeats 1 and 2 give the stability count, reported, not gated |
| Verdict | `go`: every gated check within its allowance, so swap. `inconclusive`: one item over in total, so keep the prompts and leave #1669 open for the next adapter bump. `no-go`: two or more items over, or the hard-failure gate fails, so keep the prompts and close #1669 |

**Why lenient misses are gated.** In strict mode `partial` and `unfaithful` both fail, so an
adapter that labels unfaithful sentences as `partial` scores perfectly there. #1669 swaps
lenient mode too, and only lenient misses show that mistake
(`test_lenient_gate_catches_unfaithful_labelled_partial`).

**Why errored arm 1 rows are catches.** Scored as misses, arm 1's truncated long items
would make room for arm 2 to let real invented claims through: four extra arm 2 misses
would still pass (`test_arm1_errors_do_not_make_room_for_arm2_misses`).

**Hard failures in items.** `compare` used to count validations, so seven items that each
error once in 384 validations still passed. The gate now counts items
(`test_items_that_error_once_count_toward_the_hard_failure_gate`).

**What a verdict means.** `go` means no evidence of a large regression, not proof that the
arms are equal. Because every check has to pass, equal arms reach `go` only about 45% of the
time if their errors are independent and about 75% if most are shared (planetf1's
simulation on #1721, with baseline rates of 10% strict false fails, 8% strict misses, 20%
lenient misses and 20% partial passes; it ignores domain clustering, so read it as a guide).
The one-item allowances on 34 and 20 items dominate, and doubling the invented and partial
pools would only move 45% to about 55%. That is why a result one item over is
`inconclusive`, not `no-go`.

**This is an allowance rule, with no significance test behind it.** Exact McNemar cannot add
anything here: within every allowance, p is above 0.05, one-sided or two-sided
(`test_mcnemar_cannot_fire_inside_the_allowance`). The paired 95% bootstrap CI is reported
too. It decides nothing: on these pools it would make equal arms reach `go` only 1 to 9% of
the time.
