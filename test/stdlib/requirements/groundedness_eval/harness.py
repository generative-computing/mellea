# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measurement harness for #1721: groundedness judgment prompts vs `hallucination_detection`.

Runs one arm over the labelled set (`labelled_set.jsonl`) and writes one JSON
line per (item, repeat), then compares two result files on the same items.

Arms:
    prompts  `GroundednessRequirement.validate` as it ships (arm 1).
    adapter  `find_citations` for the citation output, verdict from
             `flag_hallucinated_content` alone (arm 2), using the rule in
             `adapter_verdict`.

Usage (needs a GPU and the adapters; see README.md)::

    python -m test.stdlib.requirements.groundedness_eval.harness run \\
        --arm prompts --model ibm-granite/granite-4.0-micro --repeats 3 \\
        --out results/prompts_strict.jsonl
    python -m test.stdlib.requirements.groundedness_eval.harness run \\
        --arm adapter --model ibm-granite/granite-4.0-micro --repeats 3 \\
        --out results/adapter_strict.jsonl
    python -m test.stdlib.requirements.groundedness_eval.harness compare \\
        results/prompts_strict.jsonl results/adapter_strict.jsonl \\
        --margin 0.05 --max-hard-failure 0.02 --rule ci

The metric and comparison functions are pure and covered by unit tests; the
`run` path is the only part that needs a model.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import math
import random
import statistics
import sys
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SET_PATH = Path(__file__).with_name("labelled_set.jsonl")
LABELS = ("faithful", "partial", "unfaithful", "NA")
_SEVERITY = {"NA": 0, "faithful": 1, "partial": 2, "unfaithful": 3}


# --------------------------------------------------------------------------- data


@dataclass(frozen=True)
class Item:
    """One labelled example.

    Attributes:
        id: Stable identifier, `<domain>-<category>-<n>`.
        category: One of the categories listed in README.md.
        documents: `[{"doc_id", "text"}]` passed to both arms.
        context: Chat turns before the response, ending with the user question.
        response: The assistant response being judged.
        sentences: Gold `[{"text", "label"}]`, split exactly as the adapter
            splits the response (NLTK Punkt).
        gold: Expected verdict for `{"strict": ..., "lenient": ...}`, where
            strict means `allow_partial_support=False`.
    """

    id: str
    category: str
    documents: tuple[dict, ...]
    context: tuple[dict, ...]
    response: str
    sentences: tuple[dict, ...]
    gold: dict

    @property
    def has_invented_claim(self) -> bool:
        """True when at least one sentence is labelled `unfaithful`."""
        return any(s["label"] == "unfaithful" for s in self.sentences)


def load_items(path: Path = SET_PATH) -> list[Item]:
    """Read the labelled set.

    Args:
        path: JSONL file, one item per line.

    Returns:
        The items in file order.
    """
    items = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                items.append(
                    Item(
                        id=r["id"],
                        category=r["category"],
                        documents=tuple(r["documents"]),
                        context=tuple(r["context"]),
                        response=r["response"],
                        sentences=tuple(r["sentences"]),
                        gold=r["gold"],
                    )
                )
    return items


def gold_verdict(item: Item, allow_partial_support: bool) -> bool:
    """Expected pass/fail for an item under the given mode."""
    return bool(item.gold["lenient" if allow_partial_support else "strict"])


# ------------------------------------------------------------------- arm 2 rule


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of `text` split the way the adapter splits a response."""
    import nltk

    return list(nltk.tokenize.punkt.PunktSentenceTokenizer().span_tokenize(text))


@dataclass
class AdapterDecision:
    """Outcome of applying the #1721 arm-2 rule to adapter records.

    Attributes:
        passed: Verdict, or `None` when the result is errored.
        labels: Per-sentence label (worst overlapping record), `None` where a
            sentence has no record.
        problem: Why the result is errored, if it is.
    """

    passed: bool | None
    labels: list[str | None]
    problem: str | None = None


def adapter_verdict(
    response: str, records: Sequence[dict], allow_partial_support: bool
) -> AdapterDecision:
    """Turn `flag_hallucinated_content` records into a verdict (#1721, arm 2).

    - any `unfaithful` sentence fails the response;
    - a `partial` sentence fails it unless `allow_partial_support` is set;
    - `NA` sentences are ignored;
    - a sentence with no record, an unknown label, or offsets outside the
      response gives an errored result (`passed=None`), a hard failure.

    Args:
        response: The judged response.
        records: Adapter output records.
        allow_partial_support: Whether `partial` sentences pass.

    Returns:
        The decision with per-sentence labels.
    """
    spans = sentence_spans(response)
    labels: list[str | None] = [None] * len(spans)
    for rec in records:
        begin, end, label = (
            rec.get("response_begin"),
            rec.get("response_end"),
            rec.get("faithfulness"),
        )
        if label not in _SEVERITY:
            return AdapterDecision(None, labels, f"unknown label {label!r}")
        if (
            not isinstance(begin, int)
            or not isinstance(end, int)
            or isinstance(begin, bool)
            or isinstance(end, bool)
            or not 0 <= begin <= end <= len(response)
        ):
            return AdapterDecision(
                None, labels, f"offsets out of range: {begin}, {end}"
            )
        for i, (s_begin, s_end) in enumerate(spans):
            if begin < s_end and s_begin < end:
                current = labels[i]
                if current is None or _SEVERITY[label] > _SEVERITY[current]:
                    labels[i] = label
    missing = [i for i, lab in enumerate(labels) if lab is None]
    if missing:
        return AdapterDecision(None, labels, f"no record for sentence(s) {missing}")
    failing = {"unfaithful"} | (set() if allow_partial_support else {"partial"})
    return AdapterDecision(not any(lab in failing for lab in labels), labels)


# ------------------------------------------------------------------- running arms


@dataclass
class Outcome:
    """One validation of one item.

    Attributes:
        item_id: The item judged.
        arm: `prompts` or `adapter`.
        allow_partial_support: Mode used.
        binding: Free-text label for the backend/binding shape.
        repeat: Repeat index (0-based).
        passed: Verdict, `None` when errored.
        errored: True for a hard failure (errored result or exception).
        error: Error text for hard failures.
        reason: Reason string returned by the arm, if any.
        labels: Arm 2 per-sentence labels.
        n_citations: Number of `find_citations` records (arm 2).
        latency_s: Wall-clock seconds for the validation.
        prompt_tokens: Summed over generation calls, if the backend reports usage.
        completion_tokens: Summed over generation calls, if reported.
    """

    item_id: str
    arm: str
    allow_partial_support: bool
    binding: str
    repeat: int
    passed: bool | None
    errored: bool
    error: str | None = None
    reason: str | None = None
    labels: list[str | None] | None = None
    n_citations: int | None = None
    latency_s: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class _UsageCounter:
    """Sums `generation.usage` over the generation calls made inside a scope."""

    def __init__(self) -> None:
        self.prompt = 0
        self.completion = 0
        self.seen = False

    def as_plugin(self) -> Any:
        from mellea.plugins.decorators import hook
        from mellea.plugins.types import PluginMode

        counter = self

        @hook("generation_post_call", mode=PluginMode.AUDIT)
        async def _count(payload: Any, context: Any) -> None:
            usage = getattr(payload.model_output.generation, "usage", None)
            if usage:
                counter.seen = True
                counter.prompt += int(usage.get("prompt_tokens") or 0)
                counter.completion += int(usage.get("completion_tokens") or 0)

        return _count


async def _enter_usage_scope(counter: _UsageCounter) -> Any:
    """Register the token counter for one validation; `None` if plugins are unavailable."""
    try:
        from mellea.plugins.registry import plugin_scope

        scope = plugin_scope(counter.as_plugin())
        await scope.__aenter__()
        return scope
    except Exception:
        return None


def _contexts(item: Item) -> tuple[Any, Any, list[Any]]:
    from mellea.stdlib.components import Document, Message
    from mellea.stdlib.context import ChatContext

    docs = [Document(doc_id=d["doc_id"], text=d["text"]) for d in item.documents]
    before = ChatContext()
    for turn in item.context:
        before = before.add(Message(turn["role"], turn["content"]))
    with_response = before.add(Message("assistant", item.response))
    return before, with_response, docs


async def run_prompts_arm(
    item: Item, backend: Any, allow_partial_support: bool
) -> dict:
    """Arm 1: `GroundednessRequirement` as it ships."""
    from mellea.stdlib.requirements.rag import GroundednessRequirement

    _, with_response, docs = _contexts(item)
    req = GroundednessRequirement(
        documents=docs, allow_partial_support=allow_partial_support
    )
    result = await req.validate(backend, with_response)
    # After #1718, an errored result carries `error` and is not a verdict.
    error = getattr(result, "error", None)
    if error is not None:
        return {"passed": None, "errored": True, "error": repr(error)}
    return {"passed": result.as_bool(), "errored": False, "reason": result.reason}


async def run_adapter_arm(
    item: Item, backend: Any, allow_partial_support: bool
) -> dict:
    """Arm 2: citations for the output, verdict from `flag_hallucinated_content`."""
    from mellea.stdlib.components.intrinsic.rag import (
        find_citations,
        flag_hallucinated_content,
    )

    before, _, docs = _contexts(item)
    try:
        # Called the same way GroundednessRequirement calls find_citations.
        citations = find_citations(item.response, docs, before, backend)
        records = flag_hallucinated_content(item.response, docs, before, backend)
    except Exception as exc:  # contract errors, bad JSON, truncated output
        return {"passed": None, "errored": True, "error": repr(exc)}
    decision = adapter_verdict(item.response, records, allow_partial_support)
    return {
        "passed": decision.passed,
        "errored": decision.passed is None,
        "error": decision.problem,
        "labels": decision.labels,
        "n_citations": len(citations),
    }


ARMS: dict[str, Callable[[Item, Any, bool], Awaitable[dict]]] = {
    "prompts": run_prompts_arm,
    "adapter": run_adapter_arm,
}


async def run_arm(
    items: Iterable[Item],
    arm: str,
    backend: Any,
    *,
    allow_partial_support: bool,
    binding: str,
    repeats: int = 1,
    out: Path | None = None,
) -> list[Outcome]:
    """Validate every item `repeats` times with one arm.

    Results already present in `out` (same item, arm, mode, binding, repeat)
    are skipped, so an interrupted run can be resumed.

    Args:
        items: Items to judge.
        arm: Key of `ARMS`.
        backend: Backend passed to the arm.
        allow_partial_support: Mode.
        binding: Label recorded with each outcome.
        repeats: Validations per item, for the stability check.
        out: Optional JSONL file to append outcomes to.

    Returns:
        The outcomes produced in this call.
    """
    fn = ARMS[arm]
    done = _done_keys(out)
    results = []
    for repeat in range(repeats):
        for item in items:
            key = (item.id, arm, allow_partial_support, binding, repeat)
            if key in done:
                continue
            counter = _UsageCounter()
            scope = await _enter_usage_scope(counter)
            t0 = time.perf_counter()
            try:
                raw = await fn(item, backend, allow_partial_support)
            except Exception as exc:
                raw = {"passed": None, "errored": True, "error": repr(exc)}
            elapsed = time.perf_counter() - t0
            if scope is not None:
                await scope.__aexit__(None, None, None)
            outcome = Outcome(
                item_id=item.id,
                arm=arm,
                allow_partial_support=allow_partial_support,
                binding=binding,
                repeat=repeat,
                latency_s=round(elapsed, 4),
                prompt_tokens=counter.prompt if counter.seen else None,
                completion_tokens=counter.completion if counter.seen else None,
                **raw,
            )
            results.append(outcome)
            if out is not None:
                _append(out, outcome)
    return results


def _done_keys(out: Path | None) -> set[tuple]:
    """Keys of outcomes already in `out`, for resuming."""
    if out is None or not out.exists():
        return set()
    return {
        (o.item_id, o.arm, o.allow_partial_support, o.binding, o.repeat)
        for o in read_outcomes(out)
    }


def _append(out: Path, outcome: Outcome) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(outcome)) + "\n")


def read_outcomes(path: Path) -> list[Outcome]:
    """Read outcomes written by `run_arm`."""
    with path.open(encoding="utf-8") as f:
        return [Outcome(**json.loads(line)) for line in f if line.strip()]


# ------------------------------------------------------------------------ metrics


def _rate(k: int, n: int) -> float | None:
    return k / n if n else None


@dataclass
class ArmMetrics:
    """Per-arm numbers asked for in #1721 (primary repeat only, except stability).

    Attributes:
        n: Items scored.
        false_fails: Gold-pass items that did not pass (errored counts as a fail).
        n_gold_pass: Gold-pass items.
        misses: Items with an invented claim that passed.
        n_invented: Items with an invented claim.
        partial_misses: Strict mode only: items whose only problem is a
            `partial` sentence that passed anyway.
        n_partial_only: Strict mode only: such items.
        hard_failures: Errored validations, over all repeats.
        n_validations: All validations, over all repeats.
        unstable_items: Items whose verdict differs between repeats.
        latency_s: Median and p90 seconds per validation.
        tokens: Mean prompt and completion tokens per validation, when reported.
        label_use: Arm 2 only: gold label -> predicted label counts per sentence.
        by_category: Category -> (errors, n) for the primary repeat.
    """

    n: int
    false_fails: int
    n_gold_pass: int
    misses: int
    n_invented: int
    partial_misses: int
    n_partial_only: int
    hard_failures: int
    n_validations: int
    unstable_items: int
    latency_s: dict
    tokens: dict
    label_use: dict = field(default_factory=dict)
    by_category: dict = field(default_factory=dict)

    @property
    def false_fail_rate(self) -> float | None:
        """False fails / gold-pass items."""
        return _rate(self.false_fails, self.n_gold_pass)

    @property
    def miss_rate(self) -> float | None:
        """Misses / items with an invented claim."""
        return _rate(self.misses, self.n_invented)

    @property
    def hard_failure_rate(self) -> float | None:
        """Errored validations / all validations."""
        return _rate(self.hard_failures, self.n_validations)


def _primary(outcomes: Sequence[Outcome]) -> dict[str, Outcome]:
    """First repeat per item (the pre-registered primary verdict)."""
    best: dict[str, Outcome] = {}
    for o in outcomes:
        if o.item_id not in best or o.repeat < best[o.item_id].repeat:
            best[o.item_id] = o
    return best


def _quantile(xs: Sequence[float], q: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, math.ceil(q * len(s)) - 1)]


def arm_metrics(items: Sequence[Item], outcomes: Sequence[Outcome]) -> ArmMetrics:
    """Compute the per-arm numbers for one arm, mode and binding.

    Args:
        items: The labelled set.
        outcomes: Outcomes of one arm in one mode on one binding.

    Returns:
        The metrics.
    """
    modes = {o.allow_partial_support for o in outcomes}
    if len(modes) != 1:
        raise ValueError(f"outcomes mix modes: {modes}")
    lenient = modes.pop()
    by_id = {i.id: i for i in items}
    primary = _primary(outcomes)

    ff = n_pass = miss = n_inv = pmiss = n_ponly = 0
    by_cat: dict[str, list[int]] = {}
    label_use: dict[str, Counter] = {}
    for item_id, o in primary.items():
        item = by_id[item_id]
        expected = gold_verdict(item, lenient)
        passed = o.passed is True
        wrong = passed != expected
        cat = by_cat.setdefault(item.category, [0, 0])
        cat[0] += int(wrong)
        cat[1] += 1
        if expected:
            n_pass += 1
            ff += int(not passed)
        if item.has_invented_claim:
            n_inv += 1
            miss += int(passed)
        elif not expected:  # strict mode, only partial sentences are wrong
            n_ponly += 1
            pmiss += int(passed)
        if o.labels is not None:
            for gold_s, pred in zip(item.sentences, o.labels):
                label_use.setdefault(gold_s["label"], Counter())[str(pred)] += 1

    verdicts: dict[str, set] = {}
    for o in outcomes:
        verdicts.setdefault(o.item_id, set()).add(o.passed)
    lat = [o.latency_s for o in outcomes]
    pt = [o.prompt_tokens for o in outcomes if o.prompt_tokens is not None]
    ct = [o.completion_tokens for o in outcomes if o.completion_tokens is not None]
    return ArmMetrics(
        n=len(primary),
        false_fails=ff,
        n_gold_pass=n_pass,
        misses=miss,
        n_invented=n_inv,
        partial_misses=pmiss,
        n_partial_only=n_ponly,
        hard_failures=sum(o.errored for o in outcomes),
        n_validations=len(outcomes),
        unstable_items=sum(len(v) > 1 for v in verdicts.values()),
        latency_s={
            "median": statistics.median(lat) if lat else None,
            "p90": _quantile(lat, 0.9),
        },
        tokens={
            "prompt_mean": statistics.mean(pt) if pt else None,
            "completion_mean": statistics.mean(ct) if ct else None,
        },
        label_use={k: dict(v) for k, v in label_use.items()},
        by_category={k: tuple(v) for k, v in sorted(by_cat.items())},
    )


# --------------------------------------------------------------------- comparison


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the two discordant counts.

    Args:
        b: Pairs where only the first arm made the error.
        c: Pairs where only the second arm made the error.

    Returns:
        The p-value (1.0 when there are no discordant pairs).
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired_difference_ci(
    first: Sequence[bool],
    second: Sequence[bool],
    *,
    reps: int = 10_000,
    seed: int = 0,
    level: float = 0.95,
) -> tuple[float, float, float]:
    """Error-rate difference (second - first) with a paired bootstrap CI.

    Args:
        first: Per-item error flags for the first arm.
        second: Per-item error flags for the second arm, same item order.
        reps: Bootstrap resamples.
        seed: RNG seed.
        level: Two-sided confidence level.

    Returns:
        `(difference, lower, upper)`.
    """
    if len(first) != len(second):
        raise ValueError("arms must be scored on the same items")
    n = len(first)
    if n == 0:
        return (0.0, 0.0, 0.0)
    d = [int(s) - int(f) for f, s in zip(first, second)]
    rng = random.Random(seed)
    stats = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(reps))
    lo = stats[int((1 - level) / 2 * reps)]
    hi = stats[min(reps - 1, int((1 + level) / 2 * reps))]
    return (sum(d) / n, lo, hi)


def compare(
    items: Sequence[Item],
    first: Sequence[Outcome],
    second: Sequence[Outcome],
    *,
    margin: float,
    max_hard_failure: float,
    rule: str = "ci",
    alpha: float = 0.05,
) -> dict:
    """Apply the #1721 decision rule to arm 1 (`first`) and arm 2 (`second`).

    Both readings of "no worse" are computed and reported; `rule` picks the
    one that decides `go`:

    - `ci`: the upper end of the 95% paired bootstrap CI of
      (arm 2 rate - arm 1 rate) is at most `margin` (non-inferiority).
    - `mcnemar`: the observed difference is at most `margin`, and the exact
      McNemar test does not show arm 2 making more of this error
      (p >= `alpha` whenever arm 2 has more discordant errors).

    Args:
        items: The labelled set.
        first: Arm 1 outcomes (one mode, one binding).
        second: Arm 2 outcomes (same mode).
        margin: Pre-registered margin, as a rate.
        max_hard_failure: Pre-registered ceiling on arm 2's hard-failure rate.
        rule: `ci` or `mcnemar`, as pre-registered.
        alpha: Significance level for the `mcnemar` rule.

    Returns:
        A dict with per-arm metrics, per-error comparisons and `go`.
    """
    if rule not in ("ci", "mcnemar"):
        raise ValueError(f"unknown rule {rule!r}")
    m1, m2 = arm_metrics(items, first), arm_metrics(items, second)
    lenient = next(iter({o.allow_partial_support for o in first}))
    if {o.allow_partial_support for o in second} != {lenient}:
        raise ValueError("compare outcomes from the same allow_partial_support mode")
    p1, p2 = _primary(first), _primary(second)
    shared = [i for i in items if i.id in p1 and i.id in p2]

    def errors(kind: str, prim: dict[str, Outcome]) -> list[bool]:
        if kind == "false_fail":
            pool = [i for i in shared if gold_verdict(i, lenient)]
            return [prim[i.id].passed is not True for i in pool]
        pool = [i for i in shared if i.has_invented_claim]
        return [prim[i.id].passed is True for i in pool]

    comparison = {}
    for kind in ("false_fail", "miss"):
        e1, e2 = errors(kind, p1), errors(kind, p2)
        b = sum(x and not y for x, y in zip(e1, e2))
        c = sum(y and not x for x, y in zip(e1, e2))
        diff, lo, hi = paired_difference_ci(e1, e2)
        p = mcnemar_exact(b, c)
        by_ci = hi <= margin
        by_mcnemar = diff <= margin and not (c > b and p < alpha)
        comparison[kind] = {
            "n": len(e1),
            "arm1_rate": _rate(sum(e1), len(e1)),
            "arm2_rate": _rate(sum(e2), len(e2)),
            "only_arm1": b,
            "only_arm2": c,
            "mcnemar_p": p,
            "diff": diff,
            "ci95": (lo, hi),
            "no_worse_ci": by_ci,
            "no_worse_mcnemar": by_mcnemar,
            "no_worse": by_ci if rule == "ci" else by_mcnemar,
        }
    hard_ok = (m2.hard_failure_rate or 0.0) <= max_hard_failure
    go = (
        comparison["false_fail"]["no_worse"]
        and comparison["miss"]["no_worse"]
        and hard_ok
    )
    return {
        "allow_partial_support": lenient,
        "rule": rule,
        "margin": margin,
        "max_hard_failure": max_hard_failure,
        "arm1": asdict(m1)
        | {
            "false_fail_rate": m1.false_fail_rate,
            "miss_rate": m1.miss_rate,
            "hard_failure_rate": m1.hard_failure_rate,
        },
        "arm2": asdict(m2)
        | {
            "false_fail_rate": m2.false_fail_rate,
            "miss_rate": m2.miss_rate,
            "hard_failure_rate": m2.hard_failure_rate,
        },
        "comparison": comparison,
        "arm2_hard_failures_ok": hard_ok,
        "go": go,
    }


# ---------------------------------------------------------------------------- CLI


def _make_backend(args: argparse.Namespace) -> Any:
    if args.backend_factory:
        module, _, attr = args.backend_factory.partition(":")
        return getattr(importlib.import_module(module), attr)()
    from mellea.backends.huggingface import LocalHFBackend

    return LocalHFBackend(model_id=args.model)


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="validate the labelled set with one arm")
    run.add_argument("--arm", choices=sorted(ARMS), required=True)
    run.add_argument("--allow-partial-support", action="store_true")
    run.add_argument("--model", default="ibm-granite/granite-4.0-micro")
    run.add_argument(
        "--backend-factory",
        help="module:function returning a backend (e.g. for Granite Switch); "
        "overrides --model",
    )
    run.add_argument(
        "--binding", default="local-file", help="label stored with results"
    )
    run.add_argument("--repeats", type=int, default=3)
    run.add_argument("--set", type=Path, default=SET_PATH)
    run.add_argument("--out", type=Path, required=True)

    cmp_ = sub.add_parser("compare", help="apply the decision rule to two result files")
    cmp_.add_argument("arm1", type=Path)
    cmp_.add_argument("arm2", type=Path)
    cmp_.add_argument("--margin", type=float, required=True)
    cmp_.add_argument("--max-hard-failure", type=float, required=True)
    cmp_.add_argument("--rule", choices=["ci", "mcnemar"], required=True)
    cmp_.add_argument("--set", type=Path, default=SET_PATH)

    args = ap.parse_args(argv)
    items = load_items(args.set)
    if args.cmd == "run":
        asyncio.run(
            run_arm(
                items,
                args.arm,
                _make_backend(args),
                allow_partial_support=args.allow_partial_support,
                binding=args.binding,
                repeats=args.repeats,
                out=args.out,
            )
        )
        return 0
    report = compare(
        items,
        read_outcomes(args.arm1),
        read_outcomes(args.arm2),
        margin=args.margin,
        max_hard_failure=args.max_hard_failure,
        rule=args.rule,
    )
    json.dump(report, sys.stdout, indent=2, default=str)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
