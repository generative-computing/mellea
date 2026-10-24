# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the #1721 labelled set and measurement harness (no model)."""

import json
from collections import Counter

import pytest

from test.stdlib.requirements.groundedness_eval import harness
from test.stdlib.requirements.groundedness_eval.harness import (
    LABELS,
    Outcome,
    adapter_verdict,
    arm_metrics,
    compare,
    load_items,
    mcnemar_exact,
    paired_difference_ci,
    sentence_spans,
)

ITEMS = load_items()
EXPECTED_COUNTS = {
    "grounded": 30,
    "grounded_synthesis": 20,
    "partial": 20,
    "invented": 30,
    "non_factual": 10,
    "mixed_non_factual": 10,
    "long": 8,
}


# ------------------------------------------------------------------ labelled set


def test_set_size_and_composition():
    assert len(ITEMS) == sum(EXPECTED_COUNTS.values())
    assert Counter(i.category for i in ITEMS) == EXPECTED_COUNTS
    assert len({i.id for i in ITEMS}) == len(ITEMS)


@pytest.mark.parametrize("item", ITEMS, ids=lambda i: i.id)
def test_gold_sentences_match_the_adapter_split(item):
    """Gold labels line up one-to-one with the sentences the adapter sees."""
    spans = sentence_spans(item.response)
    assert [item.response[b:e] for b, e in spans] == [s["text"] for s in item.sentences]
    assert all(s["label"] in LABELS for s in item.sentences)


@pytest.mark.parametrize("item", ITEMS, ids=lambda i: i.id)
def test_gold_verdicts_follow_from_sentence_labels(item):
    labels = {s["label"] for s in item.sentences}
    assert item.gold["strict"] == labels.isdisjoint({"partial", "unfaithful"})
    assert item.gold["lenient"] == ("unfaithful" not in labels)
    assert item.documents and item.context[-1]["role"] == "user"


def test_categories_mean_what_they_say():
    for item in ITEMS:
        labels = [s["label"] for s in item.sentences]
        if item.category in ("grounded", "grounded_synthesis"):
            assert set(labels) == {"faithful"}
        elif item.category == "partial":
            assert "partial" in labels and "unfaithful" not in labels
        elif item.category == "invented":
            assert "unfaithful" in labels
        elif item.category == "non_factual":
            assert set(labels) == {"NA"}
        elif item.category == "mixed_non_factual":
            assert set(labels) == {"NA", "faithful"}
    longs = [i for i in ITEMS if i.category == "long"]
    assert {len(i.sentences) for i in longs} == {20, 40, 60, 80}
    assert sum(i.has_invented_claim for i in longs) == 4


# ------------------------------------------------------------------- arm 2 rule


def _records(response, labels):
    return [
        {"response_begin": b, "response_end": e, "faithfulness": lab}
        for (b, e), lab in zip(sentence_spans(response), labels)
    ]


RESPONSE = "Paris is in France. It has a river. Hope that helps!"


@pytest.mark.parametrize(
    ("labels", "lenient", "expected"),
    [
        (["faithful", "faithful", "NA"], False, True),
        (["faithful", "unfaithful", "NA"], True, False),
        (["faithful", "partial", "NA"], False, False),
        (["faithful", "partial", "NA"], True, True),
        (["NA", "NA", "NA"], False, True),
    ],
)
def test_adapter_verdict_rule(labels, lenient, expected):
    decision = adapter_verdict(RESPONSE, _records(RESPONSE, labels), lenient)
    assert decision.passed is expected
    assert decision.labels == labels


def test_sentence_without_a_record_is_a_hard_failure():
    recs = _records(RESPONSE, ["faithful", "faithful", "NA"])[:2]
    decision = adapter_verdict(RESPONSE, recs, False)
    assert decision.passed is None and "no record" in decision.problem


@pytest.mark.parametrize(
    "bad",
    [
        {"faithfulness": "hallucinated"},
        {"response_end": 10_000},
        {"response_begin": True},
        {"response_begin": 5, "response_end": 2},
    ],
)
def test_malformed_records_are_hard_failures(bad):
    recs = _records(RESPONSE, ["faithful", "faithful", "NA"])
    recs[0] = recs[0] | bad
    assert adapter_verdict(RESPONSE, recs, False).passed is None


def test_worst_overlapping_record_wins():
    b, e = sentence_spans(RESPONSE)[0]
    recs = [
        *_records(RESPONSE, ["faithful", "faithful", "NA"]),
        {"response_begin": b, "response_end": e, "faithfulness": "unfaithful"},
    ]
    assert adapter_verdict(RESPONSE, recs, True).passed is False


# ------------------------------------------------------------------- statistics


def test_mcnemar_exact_known_values():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(3, 3) == 1.0
    assert mcnemar_exact(0, 6) == pytest.approx(2 / 64)
    assert mcnemar_exact(1, 9) == pytest.approx(2 * 11 / 1024)


def test_paired_difference_ci():
    assert paired_difference_ci([True, False], [True, False]) == (0.0, 0.0, 0.0)
    diff, lo, hi = paired_difference_ci([False] * 50, [True] * 10 + [False] * 40)
    assert diff == pytest.approx(0.2) and lo <= 0.2 <= hi and lo > 0
    with pytest.raises(ValueError):
        paired_difference_ci([True], [True, False])


# ------------------------------------------------------------- metrics, decision


def _outcomes(arm, wrong_ids=(), errored_ids=(), lenient=False, repeats=1):
    out = []
    for r in range(repeats):
        for item in ITEMS:
            expected = harness.gold_verdict(item, lenient)
            if item.id in errored_ids:
                out.append(
                    Outcome(item.id, arm, lenient, "fake", r, None, True, "boom")
                )
                continue
            passed = (not expected) if item.id in wrong_ids else expected
            out.append(Outcome(item.id, arm, lenient, "fake", r, passed, False))
    return out


def test_perfect_arm_metrics():
    m = arm_metrics(ITEMS, _outcomes("adapter"))
    strict_pass = sum(i.gold["strict"] for i in ITEMS)
    assert (m.false_fails, m.n_gold_pass) == (0, strict_pass)
    assert (m.misses, m.n_invented) == (0, 34)
    assert m.n_partial_only == 20 and m.partial_misses == 0
    assert m.hard_failures == 0 and m.unstable_items == 0


def test_errors_and_instability_are_counted():
    grounded = [i.id for i in ITEMS if i.category == "grounded"]
    invented = [i.id for i in ITEMS if i.category == "invented"][:2]
    out = _outcomes(
        "prompts", wrong_ids=grounded[:3] + invented, errored_ids=grounded[3:4]
    )
    out += [o for o in _outcomes("prompts", repeats=2) if o.repeat == 1]
    m = arm_metrics(ITEMS, out)
    assert m.false_fails == 4  # three wrong + one errored grounded item
    assert m.misses == 2
    assert m.hard_failures == 1
    assert (
        m.unstable_items == 6
    )  # every item whose repeat-0 verdict differs from repeat 1
    assert m.by_category["grounded"] == (4, 30)


def test_go_when_arm2_is_no_worse():
    worse_arm1 = [i.id for i in ITEMS if i.category == "grounded"][:10]
    report = compare(
        ITEMS,
        _outcomes("prompts", wrong_ids=worse_arm1),
        _outcomes("adapter"),
        margin=0.05,
        max_hard_failure=0.02,
    )
    assert report["go"] is True
    assert report["comparison"]["false_fail"]["only_arm1"] == 10
    assert report["comparison"]["false_fail"]["mcnemar_p"] < 0.01


def test_no_go_when_arm2_misses_invented_claims():
    invented = [i.id for i in ITEMS if i.has_invented_claim][:12]
    report = compare(
        ITEMS,
        _outcomes("prompts"),
        _outcomes("adapter", wrong_ids=invented),
        margin=0.05,
        max_hard_failure=0.02,
    )
    assert report["comparison"]["miss"]["no_worse"] is False
    assert report["go"] is False


def test_no_go_on_hard_failures():
    errored = [i.id for i in ITEMS][:5]
    report = compare(
        ITEMS,
        _outcomes("prompts"),
        _outcomes("adapter", errored_ids=errored),
        margin=0.5,
        max_hard_failure=0.02,
    )
    assert report["arm2_hard_failures_ok"] is False and report["go"] is False


def test_compare_rejects_mixed_modes():
    with pytest.raises(ValueError):
        compare(
            ITEMS,
            _outcomes("prompts"),
            _outcomes("adapter", lenient=True),
            margin=0.05,
            max_hard_failure=0.02,
        )


# ------------------------------------------------------------------ run + resume


async def test_run_arm_writes_and_resumes(tmp_path, monkeypatch):
    calls = []

    async def fake_arm(item, backend, lenient):
        calls.append(item.id)
        return {"passed": True, "errored": False}

    monkeypatch.setitem(harness.ARMS, "fake", fake_arm)
    out = tmp_path / "fake.jsonl"
    items = ITEMS[:3]
    first = await harness.run_arm(
        items,
        "fake",
        None,
        allow_partial_support=False,
        binding="test",
        repeats=2,
        out=out,
    )
    assert len(first) == 6 and len(calls) == 6
    again = await harness.run_arm(
        items,
        "fake",
        None,
        allow_partial_support=False,
        binding="test",
        repeats=2,
        out=out,
    )
    assert again == [] and len(calls) == 6
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert {r["repeat"] for r in rows} == {0, 1}


async def test_arm_exceptions_become_hard_failures(monkeypatch):
    async def broken(item, backend, lenient):
        raise RuntimeError("adapter not loaded")

    monkeypatch.setitem(harness.ARMS, "broken", broken)
    (o,) = await harness.run_arm(
        ITEMS[:1], "broken", None, allow_partial_support=False, binding="t"
    )
    assert o.errored and o.passed is None and "adapter not loaded" in o.error


def test_rules_differ_when_arms_tie_with_disagreements():
    """Equal error counts, a few disagreements: mcnemar passes, the strict CI may not."""
    inv = [i.id for i in ITEMS if i.has_invented_claim]
    report = compare(
        ITEMS,
        _outcomes("prompts", wrong_ids=inv[:3]),
        _outcomes("adapter", wrong_ids=inv[3:6]),
        margin=0.05,
        max_hard_failure=0.02,
        rule="mcnemar",
    )
    miss = report["comparison"]["miss"]
    assert miss["diff"] == 0 and miss["mcnemar_p"] == 1.0
    assert miss["no_worse_mcnemar"] is True and miss["no_worse"] is True
    assert miss["no_worse_ci"] is False  # 34 items are too few for a 5-point CI
    with pytest.raises(ValueError):
        compare(
            ITEMS,
            _outcomes("prompts"),
            _outcomes("adapter"),
            margin=0.05,
            max_hard_failure=0.02,
            rule="vibes",
        )
