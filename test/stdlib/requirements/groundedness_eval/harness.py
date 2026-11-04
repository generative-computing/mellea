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
        --margin 0.05 --max-hard-failure 0.02 --rule margin

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
import warnings
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
        error_type: Exception class name, or `malformed_records` when the adapter
            output parsed but `adapter_verdict` could not use it.
        reason: Reason string returned by the arm, if any.
        labels: Arm 2 per-sentence labels.
        n_citations: Number of `find_citations` records (arm 2).
        latency_s: Wall-clock seconds for the validation.
        prompt_tokens: Summed over generation calls, or `None` unless every call
            in the validation reported usage, so a partial total never looks real.
        completion_tokens: Same as `prompt_tokens`.
        generation_calls: Generation calls seen, `None` without the `hooks` extra.
        generation_calls_with_usage: How many of those reported usage.
        model: The `--model` value the run was started with.
        adapters: `{"name", "revision", "binding_type"}` for each adapter that
            ran, `None` without the `hooks` extra.
    """

    item_id: str
    arm: str
    allow_partial_support: bool
    binding: str
    repeat: int
    passed: bool | None
    errored: bool
    error: str | None = None
    error_type: str | None = None
    reason: str | None = None
    labels: list[str | None] | None = None
    n_citations: int | None = None
    latency_s: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    generation_calls: int | None = None
    generation_calls_with_usage: int | None = None
    model: str | None = None
    adapters: list[dict] | None = None


class _CallRecorder:
    """Records the generation and adapter calls made inside one validation."""

    def __init__(self) -> None:
        self.active = False
        self.calls = 0
        self.calls_with_usage = 0
        self.prompt = 0
        self.completion = 0
        self._adapters: set[tuple[str, str | None, str]] = set()

    def note_generation(self, usage: dict | None) -> None:
        """Count one generation call and add its usage, if it reported any."""
        self.calls += 1
        if usage:
            self.calls_with_usage += 1
            self.prompt += int(usage.get("prompt_tokens") or 0)
            self.completion += int(usage.get("completion_tokens") or 0)

    def note_adapter(self, name: str, revision: str | None, binding_type: str) -> None:
        """Record which adapter ran, at which revision and on which binding."""
        self._adapters.add((name, revision, binding_type))

    def fields(self) -> dict:
        """The `Outcome` fields this recorder fills in."""
        if not self.active:
            return {}
        complete = self.calls > 0 and self.calls == self.calls_with_usage
        return {
            "prompt_tokens": self.prompt if complete else None,
            "completion_tokens": self.completion if complete else None,
            "generation_calls": self.calls,
            "generation_calls_with_usage": self.calls_with_usage,
            "adapters": [
                {"name": n, "revision": r, "binding_type": b}
                for n, r, b in sorted(self._adapters, key=str)
            ],
        }

    def as_plugins(self) -> list[Any]:
        """The scoped hooks that feed this recorder."""
        from mellea.plugins.decorators import hook
        from mellea.plugins.types import PluginMode

        recorder = self

        @hook("generation_post_call", mode=PluginMode.AUDIT)
        async def _generation(payload: Any, context: Any) -> None:
            generation = getattr(payload.model_output, "generation", None)
            recorder.note_generation(getattr(generation, "usage", None))

        @hook("adapter_function_invocation_complete", mode=PluginMode.AUDIT)
        async def _adapter(payload: Any, context: Any) -> None:
            recorder.note_adapter(payload.name, payload.revision, payload.binding_type)

        return [_generation, _adapter]


async def _enter_recording_scope(recorder: _CallRecorder) -> Any:
    """Register the recorder for one validation; `None` without the `hooks` extra."""
    try:
        from mellea.plugins.registry import plugin_scope

        scope = plugin_scope(*recorder.as_plugins())
        await scope.__aenter__()
    except ImportError:
        warnings.warn(
            "the `hooks` extra is not installed, so tokens and adapter revisions "
            "will not be recorded: pip install 'mellea[hooks]'",
            stacklevel=2,
        )
        return None
    recorder.active = True
    return scope


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
        return {
            "passed": None,
            "errored": True,
            "error": repr(error),
            "error_type": type(error).__name__,
        }
    return {"passed": result.as_bool(), "errored": False, "reason": result.reason}


async def run_adapter_arm(
    item: Item, backend: Any, allow_partial_support: bool
) -> dict:
    """Arm 2: citations for the output, verdict from `flag_hallucinated_content`.

    Only the errors the adapter wrappers raise for bad model output become hard
    failures. Anything else, such as running out of memory, propagates and stops
    the run. The adapters are resolved outside the `try` because a failed download
    can raise a `ValueError`, and that says nothing about the adapter's output.
    """
    from mellea.backends.adapters import AdapterSchemaMismatchError
    from mellea.stdlib.components.intrinsic.rag import (
        find_citations,
        flag_hallucinated_content,
    )

    before, _, docs = _contexts(item)
    backend.resolve_adapter("citations")
    backend.resolve_adapter("hallucination_detection")
    try:
        # Called the same way GroundednessRequirement calls find_citations.
        citations = find_citations(item.response, docs, before, backend)
        records = flag_hallucinated_content(item.response, docs, before, backend)
    except (ValueError, AdapterSchemaMismatchError) as exc:
        return {
            "passed": None,
            "errored": True,
            "error": repr(exc),
            "error_type": type(exc).__name__,
        }
    decision = adapter_verdict(item.response, records, allow_partial_support)
    return {
        "passed": decision.passed,
        "errored": decision.passed is None,
        "error": decision.problem,
        "error_type": None if decision.passed is not None else "malformed_records",
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
    model: str | None = None,
) -> list[Outcome]:
    """Validate every item `repeats` times with one arm.

    Results already present in `out` (same item, arm, mode, binding, repeat,
    model) are skipped, so an interrupted run can be resumed. An exception the
    arm does not turn into a result stops the run and writes no row for that
    item, so the item runs again on resume. Recorded hard failures stay done:
    retrying them would let a flaky failure pass the hard-failure gate.

    Args:
        items: Items to judge.
        arm: Key of `ARMS`.
        backend: Backend passed to the arm.
        allow_partial_support: Mode.
        binding: Label recorded with each outcome.
        repeats: Validations per item, for the stability check.
        out: Optional JSONL file to append outcomes to.
        model: Base model label recorded with each outcome.

    Returns:
        The outcomes produced in this call.
    """
    fn = ARMS[arm]
    done = _done_keys(out)
    results = []
    for repeat in range(repeats):
        for item in items:
            key = (item.id, arm, allow_partial_support, binding, repeat, model)
            if key in done:
                continue
            recorder = _CallRecorder()
            scope = await _enter_recording_scope(recorder)
            t0 = time.perf_counter()
            try:
                raw = await fn(item, backend, allow_partial_support)
                elapsed = time.perf_counter() - t0
            finally:
                if scope is not None:
                    await scope.__aexit__(None, None, None)
            outcome = Outcome(
                item_id=item.id,
                arm=arm,
                allow_partial_support=allow_partial_support,
                binding=binding,
                repeat=repeat,
                latency_s=round(elapsed, 4),
                model=model,
                **recorder.fields(),
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
        (o.item_id, o.arm, o.allow_partial_support, o.binding, o.repeat, o.model)
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
        tokens: Mean prompt and completion tokens over the validations with a
            complete total, and how many validations that was.
        adapters: Each distinct adapter name, revision and binding type that ran.
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
    adapters: list[dict] = field(default_factory=list)
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


def _adapters_used(outcomes: Iterable[Outcome]) -> list[dict]:
    """The distinct adapter records across `outcomes`."""
    seen = {
        (a["name"], a["revision"], a["binding_type"])
        for o in outcomes
        for a in o.adapters or ()
    }
    return [
        {"name": n, "revision": r, "binding_type": b}
        for n, r, b in sorted(seen, key=str)
    ]


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
            "validations_with_totals": len(pt),
        },
        adapters=_adapters_used(outcomes),
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


def _check_inputs(
    items: Sequence[Item], first: Sequence[Outcome], second: Sequence[Outcome]
) -> None:
    """Reject result files that cannot give a meaningful comparison.

    Raises:
        ValueError: If either file is empty, mixes arms, bindings, modes or
            models, holds the wrong arm, has duplicate rows, has unknown items,
            lacks an item at repeat 0, or stops partway through a repeat; if the
            files differ in mode or model; or if an adapter ran at two revisions.
    """
    known = {i.id for i in items}
    for expected_arm, outcomes in (("prompts", first), ("adapter", second)):
        if not outcomes:
            raise ValueError(f"no outcomes for the {expected_arm} arm")
        for attr, plural in (
            ("arm", "arms"),
            ("binding", "bindings"),
            ("allow_partial_support", "modes"),
            ("model", "models"),
        ):
            values = {getattr(o, attr) for o in outcomes}
            if len(values) != 1:
                raise ValueError(
                    f"outcomes mix {plural}: {sorted(values, key=str)}; "
                    "compare one run per file"
                )
        if outcomes[0].arm != expected_arm:
            raise ValueError(
                f"expected the {expected_arm} arm here, got {outcomes[0].arm!r}: "
                "pass the prompts file first and the adapter file second"
            )
        keys = Counter((o.item_id, o.repeat) for o in outcomes)
        duplicates = sorted(k for k, n in keys.items() if n > 1)
        if duplicates:
            raise ValueError(
                f"{expected_arm}: {len(duplicates)} duplicate rows, e.g. {duplicates[0]}"
            )
        repeats: dict[str, set[int]] = {}
        for o in outcomes:
            repeats.setdefault(o.item_id, set()).add(o.repeat)
        unknown = sorted(repeats.keys() - known)
        if unknown:
            raise ValueError(
                f"{expected_arm}: {len(unknown)} items not in the set, e.g. {unknown[0]}"
            )
        missing = sorted(i for i in known if 0 not in repeats.get(i, ()))
        if missing:
            raise ValueError(
                f"{expected_arm}: {len(missing)} of {len(known)} items missing "
                f"at repeat 0, e.g. {missing[0]}"
            )
        if len({frozenset(r) for r in repeats.values()}) != 1:
            raise ValueError(
                f"{expected_arm}: items have different repeats; finish the run first"
            )
    if first[0].allow_partial_support != second[0].allow_partial_support:
        raise ValueError("compare outcomes from the same allow_partial_support mode")
    if first[0].model != second[0].model:
        raise ValueError(
            f"compare outcomes from the same model: "
            f"{first[0].model!r} vs {second[0].model!r}"
        )
    revisions: dict[str, set[str | None]] = {}
    for a in _adapters_used([*first, *second]):
        revisions.setdefault(a["name"], set()).add(a["revision"])
    mixed = {n: sorted(r, key=str) for n, r in revisions.items() if len(r) > 1}
    if mixed:
        raise ValueError(f"outcomes mix adapter revisions: {mixed}")


def compare(
    items: Sequence[Item],
    first: Sequence[Outcome],
    second: Sequence[Outcome],
    *,
    margin: float,
    max_hard_failure: float,
    rule: str = "margin",
) -> dict:
    """Apply the #1721 decision rule to arm 1 (`first`) and arm 2 (`second`).

    Both readings of "no worse" are computed and reported; `rule` picks the
    one that decides `go`:

    - `margin`: the observed difference (arm 2 rate - arm 1 rate) is at most
      `margin`.
    - `ci`: the upper end of the 95% paired bootstrap CI of that difference is
      at most `margin` (non-inferiority).

    The exact McNemar p-value is reported for information and decides nothing.

    The hard-failure ceiling applies per binding, so call this once per binding.

    Args:
        items: The labelled set.
        first: Arm 1 outcomes (one run: one mode, binding and model).
        second: Arm 2 outcomes (one run, same mode and model).
        margin: Pre-registered margin, as a rate.
        max_hard_failure: Pre-registered ceiling on arm 2's hard-failure rate.
        rule: `margin` or `ci`, as pre-registered.

    Returns:
        A dict with per-arm metrics, per-error comparisons and `go`.

    Raises:
        ValueError: If `rule` is unknown or the outcomes fail `_check_inputs`.
    """
    if rule not in ("margin", "ci"):
        raise ValueError(f"unknown rule {rule!r}")
    _check_inputs(items, first, second)
    lenient = first[0].allow_partial_support
    m1, m2 = arm_metrics(items, first), arm_metrics(items, second)
    p1, p2 = _primary(first), _primary(second)
    shared = list(items)

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
        by_ci = hi <= margin
        by_margin = diff <= margin
        comparison[kind] = {
            "n": len(e1),
            "arm1_rate": _rate(sum(e1), len(e1)),
            "arm2_rate": _rate(sum(e2), len(e2)),
            "only_arm1": b,
            "only_arm2": c,
            "mcnemar_p": mcnemar_exact(b, c),
            "diff": diff,
            "ci95": (lo, hi),
            "no_worse_ci": by_ci,
            "no_worse_margin": by_margin,
            "no_worse": by_ci if rule == "ci" else by_margin,
        }
    hard_ok = (m2.hard_failure_rate or 0.0) <= max_hard_failure
    go = (
        comparison["false_fail"]["no_worse"]
        and comparison["miss"]["no_worse"]
        and hard_ok
    )
    return {
        "allow_partial_support": lenient,
        "model": first[0].model,
        "bindings": {"arm1": first[0].binding, "arm2": second[0].binding},
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
    run.add_argument(
        "--model",
        default="ibm-granite/granite-4.0-micro",
        help="base model, stored with results",
    )
    run.add_argument(
        "--backend-factory",
        help="module:function returning a backend (e.g. for Granite Switch); "
        "--model then only labels the results, so set it to the factory's base model",
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
    cmp_.add_argument("--rule", choices=["margin", "ci"], required=True)
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
                model=args.model,
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
