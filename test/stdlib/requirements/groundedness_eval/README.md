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
    errored. An exception from either adapter call (a contract error after #1719, bad
    JSON, truncated output) is also errored. Errored counts as a fail and as a hard failure.

Run each arm in both modes. Arm 2 runs on both binding shapes (`--binding` labels the
results; `--backend-factory module:function` supplies a non-default backend such as
Granite Switch).

## What is measured (per arm, mode and binding)

- False fails: gold-pass items that did not pass. This decides the swap.
- Misses: items with an invented claim that passed.
- Strict mode only: partial-only items that passed.
- Label use (arm 2): gold sentence label against adapter label, so it shows whether
  `partial` and `NA` are used where they apply.
- Hard failures over all repeats.
- Stability: items whose verdict changes across repeats (`--repeats 3`, temperature 0).
- Latency (median, p90) and tokens per validation. Tokens are summed from
  `generation.usage` through a scoped plugin, and are left empty if a backend does not
  report usage.

The first repeat is the primary verdict for every comparison. The other repeats only
feed the stability count.

## Running (GPU)

```bash
H="python -m test.stdlib.requirements.groundedness_eval.harness"
$H run --arm prompts --out results/prompts_strict.jsonl
$H run --arm prompts --allow-partial-support --out results/prompts_lenient.jsonl
$H run --arm adapter --out results/adapter_strict.jsonl
$H run --arm adapter --allow-partial-support --out results/adapter_lenient.jsonl
$H compare results/prompts_strict.jsonl results/adapter_strict.jsonl \
    --margin 0.05 --max-hard-failure 0.02 --rule mcnemar
```

Runs append to `--out` and skip finished items, so an interrupted run resumes.

## Decision rule: proposal, to be agreed on #1721 before any run

| | Proposed |
|---|---|
| Set | the 128 items above |
| Primary verdict | repeat 0 of 3 |
| Margin | 5 percentage points, for false fails and for misses, in each mode |
| Paired test | exact McNemar on the items where the arms disagree, α = 0.05 |
| "No worse" | rule `mcnemar`: arm 2's rate is at most arm 1's + 5 points, and McNemar does not show arm 2 making more of that error |
| Hard failures | arm 2 under 2% of validations, on each binding |

**Why `mcnemar`, not `ci`.** The harness also computes a strict non-inferiority
reading (`--rule ci`: the upper end of the 95% paired bootstrap CI must be within the
margin). On this set that reading is too demanding. Suppose the two arms each miss
3 of the 34 invented-claim items, but different ones. Their miss rates are equal, yet
the CI is wide enough to call arm 2 worse. `test_rules_differ_when_arms_tie_with_disagreements`
reproduces this. Making `ci` workable at a 5-point margin would need a much larger
set. Both readings are always reported, so the choice stays visible.
