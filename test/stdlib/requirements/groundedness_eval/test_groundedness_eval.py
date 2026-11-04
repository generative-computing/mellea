# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the #1721 labelled set and measurement harness (no model)."""

import json
from collections import Counter
from dataclasses import replace

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
    # The adapter's offsets assume single spaces between sentences.
    assert " ".join(s["text"] for s in item.sentences) == item.response
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


@pytest.mark.parametrize(
    ("n", "extra_errors"),
    [(34, 1), (74, 3), (94, 4)],
    ids=["miss", "strict", "lenient"],
)
def test_mcnemar_cannot_fire_inside_the_margin(n, extra_errors):
    """On this set, the rule is the 5-point margin and nothing more."""
    largest = 0
    for b in range(n + 1):
        for c in range(b + 1, n - b + 1):
            if (c - b) / n <= 0.05:
                largest = max(largest, c - b)
                two_sided = mcnemar_exact(b, c)
                assert two_sided >= 0.05 and two_sided / 2 >= 0.05
    assert largest == extra_errors


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


async def test_unexpected_exceptions_stop_the_run_and_retry_on_resume(
    tmp_path, monkeypatch
):
    calls = []

    async def flaky(item, backend, lenient):
        calls.append(item.id)
        if len(calls) == 2:
            raise RuntimeError("CUDA out of memory")
        return {"passed": True, "errored": False}

    monkeypatch.setitem(harness.ARMS, "flaky", flaky)
    out = tmp_path / "flaky.jsonl"
    kwargs = {"allow_partial_support": False, "binding": "t", "out": out}
    with pytest.raises(RuntimeError):
        await harness.run_arm(ITEMS[:3], "flaky", None, **kwargs)
    assert [r.item_id for r in harness.read_outcomes(out)] == [ITEMS[0].id]
    await harness.run_arm(ITEMS[:3], "flaky", None, **kwargs)
    assert [r.item_id for r in harness.read_outcomes(out)] == [i.id for i in ITEMS[:3]]


async def test_recorded_hard_failures_are_not_retried(tmp_path, monkeypatch):
    calls = []

    async def bad_output(item, backend, lenient):
        calls.append(item.id)
        return {"passed": None, "errored": True, "error": "x", "error_type": "E"}

    monkeypatch.setitem(harness.ARMS, "bad", bad_output)
    kwargs = {
        "allow_partial_support": False,
        "binding": "t",
        "out": tmp_path / "bad.jsonl",
    }
    await harness.run_arm(ITEMS[:1], "bad", None, **kwargs)
    assert await harness.run_arm(ITEMS[:1], "bad", None, **kwargs) == []
    assert len(calls) == 1


async def test_a_different_model_is_not_skipped_on_resume(tmp_path, monkeypatch):
    async def fake(item, backend, lenient):
        return {"passed": True, "errored": False}

    monkeypatch.setitem(harness.ARMS, "fake", fake)
    kwargs = {
        "allow_partial_support": False,
        "binding": "t",
        "out": tmp_path / "m.jsonl",
    }
    await harness.run_arm(ITEMS[:1], "fake", None, model="a", **kwargs)
    (o,) = await harness.run_arm(ITEMS[:1], "fake", None, model="b", **kwargs)
    assert o.model == "b"


class _Backend:
    def __init__(self, resolve_error=None):
        self.resolve_error = resolve_error

    def resolve_adapter(self, name):
        if self.resolve_error:
            raise self.resolve_error


def _patch_adapters(monkeypatch, *, citations=None, flagged=None):
    from mellea.stdlib.components.intrinsic import rag

    def fake(result):
        def call(*args, **kwargs):
            if isinstance(result, Exception):
                raise result
            return result

        return call

    monkeypatch.setattr(rag, "find_citations", fake(citations or []))
    monkeypatch.setattr(rag, "flag_hallucinated_content", fake(flagged or []))


@pytest.mark.parametrize("name", ["value", "contract"])
async def test_adapter_output_errors_are_hard_failures(monkeypatch, name):
    from mellea.backends.adapters import AdapterSchemaMismatchError

    exc = (
        ValueError("bad JSON")
        if name == "value"
        else AdapterSchemaMismatchError("x", frozenset(), frozenset({"y"}))
    )
    _patch_adapters(monkeypatch, flagged=exc)
    raw = await harness.run_adapter_arm(ITEMS[0], _Backend(), False)
    assert raw["errored"] and raw["passed"] is None
    assert raw["error_type"] == type(exc).__name__


async def test_other_adapter_arm_errors_propagate(monkeypatch):
    _patch_adapters(monkeypatch, flagged=RuntimeError("CUDA out of memory"))
    with pytest.raises(RuntimeError):
        await harness.run_adapter_arm(ITEMS[0], _Backend(), False)


async def test_failed_adapter_resolve_stops_the_run(monkeypatch):
    """A download failure can be a ValueError; it must not count against the adapter."""
    _patch_adapters(monkeypatch)
    with pytest.raises(ValueError):
        await harness.run_adapter_arm(
            ITEMS[0], _Backend(resolve_error=ValueError("no model id")), False
        )


async def test_unusable_records_name_their_error_type(monkeypatch):
    _patch_adapters(monkeypatch)  # no records, so every sentence lacks one
    raw = await harness.run_adapter_arm(ITEMS[0], _Backend(), False)
    assert raw["errored"] and raw["error_type"] == "malformed_records"


# ------------------------------------------------------------------- call recording


def test_partial_token_totals_are_not_stored():
    rec = harness._CallRecorder()
    rec.active = True
    rec.note_generation({"prompt_tokens": 10, "completion_tokens": 2})
    rec.note_generation(None)  # the HF adapter path reports no usage
    fields = rec.fields()
    assert fields["prompt_tokens"] is None and fields["completion_tokens"] is None
    assert (fields["generation_calls"], fields["generation_calls_with_usage"]) == (2, 1)


def test_complete_token_totals_and_adapters_are_stored():
    rec = harness._CallRecorder()
    rec.active = True
    rec.note_generation({"prompt_tokens": 10, "completion_tokens": 2})
    rec.note_generation({"prompt_tokens": 5, "completion_tokens": 1})
    rec.note_adapter("citations", "abc", "local_file")
    rec.note_adapter("citations", "abc", "local_file")
    fields = rec.fields()
    assert (fields["prompt_tokens"], fields["completion_tokens"]) == (15, 3)
    assert fields["adapters"] == [
        {"name": "citations", "revision": "abc", "binding_type": "local_file"}
    ]


def test_no_generation_calls_means_no_token_total():
    rec = harness._CallRecorder()
    rec.active = True
    assert rec.fields()["prompt_tokens"] is None


async def test_scoped_hooks_fill_in_the_row(monkeypatch):
    pytest.importorskip("cpex")
    from types import SimpleNamespace

    from mellea.plugins.hooks.adapter_function import (
        AdapterFunctionInvocationCompletePayload,
    )
    from mellea.plugins.hooks.generation import GenerationPostCallPayload
    from mellea.plugins.manager import invoke_hook
    from mellea.plugins.types import HookType

    async def fake(item, backend, lenient):
        output = SimpleNamespace(
            generation=SimpleNamespace(
                usage={"prompt_tokens": 7, "completion_tokens": 3}
            )
        )
        await invoke_hook(
            HookType.GENERATION_POST_CALL,
            GenerationPostCallPayload(model_output=output),
        )
        await invoke_hook(
            HookType.ADAPTER_FUNCTION_INVOCATION_COMPLETE,
            AdapterFunctionInvocationCompletePayload(
                name="citations",
                revision="abc",
                binding_type="local_file",
                adapter_type="lora",
                outcome="success",
            ),
        )
        return {"passed": True, "errored": False}

    monkeypatch.setitem(harness.ARMS, "fake", fake)
    (o,) = await harness.run_arm(
        ITEMS[:1], "fake", None, allow_partial_support=False, binding="t", model="m"
    )
    assert (o.prompt_tokens, o.completion_tokens) == (7, 3)
    assert o.generation_calls == o.generation_calls_with_usage == 1
    assert o.adapters == [
        {"name": "citations", "revision": "abc", "binding_type": "local_file"}
    ]
    assert o.model == "m"


# ---------------------------------------------------------------- compare's inputs


def _compare(first, second, **kwargs):
    return compare(ITEMS, first, second, margin=0.05, max_hard_failure=0.02, **kwargs)


def test_compare_rejects_missing_items():
    with pytest.raises(ValueError, match="missing"):
        _compare(_outcomes("prompts"), _outcomes("adapter")[:10])


def test_compare_rejects_swapped_files():
    with pytest.raises(ValueError, match="prompts file first"):
        _compare(_outcomes("adapter"), _outcomes("prompts"))


def test_compare_rejects_mixed_bindings():
    """Clean rows must not dilute another binding's hard failures below the gate."""
    other = [replace(o, binding="switch") for o in _outcomes("adapter")]
    with pytest.raises(ValueError, match="mix bindings"):
        _compare(_outcomes("prompts"), _outcomes("adapter") + other)


def test_compare_rejects_mixed_arms():
    with pytest.raises(ValueError, match="mix arms"):
        _compare(_outcomes("prompts") + _outcomes("adapter"), _outcomes("adapter"))


def test_compare_rejects_duplicate_rows():
    rows = _outcomes("adapter")
    with pytest.raises(ValueError, match="duplicate"):
        _compare(_outcomes("prompts"), rows + rows[:1])


def test_compare_rejects_items_outside_the_set():
    rows = _outcomes("adapter")
    with pytest.raises(ValueError, match="not in the set"):
        _compare(_outcomes("prompts"), [*rows, replace(rows[0], item_id="nope")])


def test_compare_rejects_an_unfinished_repeat():
    rows = _outcomes("adapter", repeats=2)
    with pytest.raises(ValueError, match="different repeats"):
        _compare(_outcomes("prompts"), rows[:-1])


def test_compare_rejects_an_empty_file():
    with pytest.raises(ValueError, match="no outcomes"):
        _compare(_outcomes("prompts"), [])


def test_compare_rejects_different_models():
    second = [replace(o, model="other") for o in _outcomes("adapter")]
    with pytest.raises(ValueError, match="same model"):
        _compare(_outcomes("prompts"), second)


def test_compare_rejects_mixed_adapter_revisions():
    def at(revision):
        return [{"name": "citations", "revision": revision, "binding_type": "x"}]

    rows = _outcomes("adapter")
    second = [replace(rows[0], adapters=at("a")), replace(rows[1], adapters=at("b"))]
    with pytest.raises(ValueError, match="adapter revisions"):
        _compare(_outcomes("prompts"), second + rows[2:])


def test_report_lists_the_adapters_and_model():
    used = [{"name": "citations", "revision": "a", "binding_type": "local_file"}]
    second = [replace(o, adapters=used, model="m") for o in _outcomes("adapter")]
    first = [replace(o, model="m") for o in _outcomes("prompts")]
    report = _compare(first, second)
    assert report["arm2"]["adapters"] == used and report["model"] == "m"


def test_margin_and_ci_readings_differ_when_arms_tie_with_disagreements():
    """Equal error counts, a few disagreements: the margin passes, the CI does not."""
    inv = [i.id for i in ITEMS if i.has_invented_claim]
    report = _compare(
        _outcomes("prompts", wrong_ids=inv[:3]),
        _outcomes("adapter", wrong_ids=inv[3:6]),
        rule="margin",
    )
    miss = report["comparison"]["miss"]
    assert miss["diff"] == 0 and miss["mcnemar_p"] == 1.0
    assert miss["no_worse_margin"] is True and miss["no_worse"] is True
    assert miss["no_worse_ci"] is False  # 34 items are too few for a 5-point CI
    with pytest.raises(ValueError):
        _compare(_outcomes("prompts"), _outcomes("adapter"), rule="vibes")


@pytest.mark.parametrize(
    ("extra_misses", "go"), [(1, True), (2, False)], ids=["one_more", "two_more"]
)
def test_margin_allows_one_more_miss_of_34(extra_misses, go):
    inv = [i.id for i in ITEMS if i.has_invented_claim]
    report = _compare(
        _outcomes("prompts"), _outcomes("adapter", wrong_ids=inv[:extra_misses])
    )
    assert report["comparison"]["miss"]["n"] == 34
    assert report["go"] is go
