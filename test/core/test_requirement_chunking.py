# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for `Requirement`'s streaming-validation chunking."""

import random
from copy import copy
from itertools import pairwise

import pytest

from mellea.core import PartialValidationResult, PartialValidationSummary, Requirement
from mellea.core.chunking import Chunker, SentenceChunking


class RecordingReq(Requirement):
    """Requirement that records every chunk its `_stream_validate` receives."""

    def __init__(self, chunking=None) -> None:
        super().__init__(description="records units", chunking=chunking)
        self.seen: list[str] = []

    async def _stream_validate(self, chunk, *, backend, ctx) -> PartialValidationResult:
        self.seen.append(chunk)
        return PartialValidationResult("unknown")


# ---------------------------------------------------------------------------
# Construction / chunking field
# ---------------------------------------------------------------------------


def test_chunking_defaults_to_none():
    assert Requirement().chunking is None


def test_chunking_accepts_strategy_instance():
    strat = SentenceChunking()
    assert Requirement(chunking=strat).chunking is strat


def test_chunking_resolves_alias_string():
    assert isinstance(Requirement(chunking="sentence").chunking, SentenceChunking)


def test_chunking_bad_alias_raises_at_construction():
    with pytest.raises(ValueError):
        Requirement(chunking="not-a-strategy")


# ---------------------------------------------------------------------------
# stream_validate driver
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_passthrough_when_chunking_none():
    """chunking=None -> the raw chunk is validated as-is, wrapped in a 1-element list."""
    req = RecordingReq()
    out = await req.stream_validate("hello world", backend=None, ctx=None)  # type: ignore[arg-type]
    assert len(out) == 1
    assert out[0].success == "unknown"
    assert req.seen == ["hello world"]


@pytest.mark.asyncio
async def test_rechunks_into_requirements_own_units():
    req = RecordingReq(chunking="sentence")
    out = await req.stream_validate("One. Two. Thre", backend=None, ctx=None)  # type: ignore[arg-type]  # codespell:ignore
    assert [r.success for r in out] == ["unknown", "unknown"]
    assert req.seen == ["One.", "Two."]  # trailing partial sentence withheld

    out2 = await req.stream_validate("e. Four", backend=None, ctx=None)  # type: ignore[arg-type]
    assert len(out2) == 1
    assert req.seen == ["One.", "Two.", "Three."]  # residual completed across the seam


@pytest.mark.asyncio
async def test_unknown_when_no_chunk_completes():
    """When the delta completes no chunk, the result is a single 'unknown' (never empty)."""
    req = RecordingReq(chunking="sentence")
    out = await req.stream_validate("no boundary yet", backend=None, ctx=None)  # type: ignore[arg-type]
    assert [r.success for r in out] == ["unknown"]
    assert req.seen == []  # _stream_validate not called — no chunk to validate


@pytest.mark.asyncio
async def test_short_circuits_at_first_failing_unit():
    """Units after the first failing unit in a chunk are not validated."""

    class FailOnBad(Requirement):
        def __init__(self) -> None:
            super().__init__(chunking="sentence")
            self.seen: list[str] = []

        async def _stream_validate(
            self, chunk, *, backend, ctx
        ) -> PartialValidationResult:
            self.seen.append(chunk)
            if "bad" in chunk:
                return PartialValidationResult("fail")
            return PartialValidationResult("unknown")

    req = FailOnBad()
    # Trailing space makes all three sentences complete units; the middle one fails.
    out = await req.stream_validate(
        "Good one. This is bad. Never seen. ",
        backend=None,  # type: ignore[arg-type]
        ctx=None,  # type: ignore[arg-type]
    )
    assert [r.success for r in out] == ["unknown", "fail"]
    assert req.seen == ["Good one.", "This is bad."]  # third sentence never validated


@pytest.mark.asyncio
async def test_stream_validate_flush_includes_residual():
    """stream_validate(flush=True) also validates the trailing residual, in the same call."""
    req = RecordingReq(chunking="sentence")
    out = await req.stream_validate("One. Tw", backend=None, ctx=None, flush=True)  # type: ignore[arg-type]
    # "One." completes; "Tw" is the residual, validated because flush=True.
    assert req.seen == ["One.", "Tw"]
    assert [r.success for r in out] == ["unknown", "unknown"]


@pytest.mark.asyncio
async def test_flush_skipped_after_mid_stream_fail():
    """flush=True does not validate the residual once an earlier chunk has failed."""

    class FailOnBad(Requirement):
        def __init__(self) -> None:
            super().__init__(chunking="sentence")
            self.seen: list[str] = []

        async def _stream_validate(
            self, chunk, *, backend, ctx
        ) -> PartialValidationResult:
            self.seen.append(chunk)
            if "bad" in chunk:
                return PartialValidationResult("fail")
            return PartialValidationResult("unknown")

    req = FailOnBad()
    # "This is bad." fails; "Resid" is the withheld residual and must not be flushed.
    out = await req.stream_validate(
        "Good one. This is bad. Resid",
        backend=None,  # type: ignore[arg-type]
        ctx=None,  # type: ignore[arg-type]
        flush=True,
    )
    assert [r.success for r in out] == ["unknown", "fail"]
    assert req.seen == [
        "Good one.",
        "This is bad.",
    ]  # residual not validated after fail


@pytest.mark.asyncio
async def test_stream_validate_empty_delta_is_unknown_without_calling_hook():
    """chunking=None returns 'unknown' for an empty delta, without calling _stream_validate."""
    req = RecordingReq()  # chunking=None (passthrough)
    out = await req.stream_validate("", backend=None, ctx=None, flush=True)  # type: ignore[arg-type]
    assert [r.success for r in out] == ["unknown"]
    assert req.seen == []  # _stream_validate not called with ""


# ---------------------------------------------------------------------------
# stream_flush
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_flush_drains_residual():
    req = RecordingReq(chunking="sentence")
    await req.stream_validate("One. Tw", backend=None, ctx=None)  # type: ignore[arg-type]
    flushed = await req.stream_flush(backend=None, ctx=None)  # type: ignore[arg-type]
    assert len(flushed) == 1
    assert req.seen == ["One.", "Tw"]


@pytest.mark.asyncio
async def test_stream_flush_empty_when_no_chunker():
    """chunking=None never builds a chunker, so there is nothing to flush."""
    req = RecordingReq()
    assert await req.stream_flush(backend=None, ctx=None) == []  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_stream_flush_empty_when_no_residual():
    req = RecordingReq(chunking="sentence")
    await req.stream_validate("One. ", backend=None, ctx=None)  # type: ignore[arg-type]
    assert await req.stream_flush(backend=None, ctx=None) == []  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# __copy__ isolation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_copy_resets_chunker():
    req = RecordingReq(chunking="sentence")
    await req.stream_validate("One. Tw", backend=None, ctx=None)  # type: ignore[arg-type]
    assert req._chunker is not None

    clone = copy(req)
    assert clone._chunker is None
    assert req._chunker is not None

    # A fresh chunker must not carry the original's "Tw" residual.
    out = await clone.stream_validate("Zeta. ", backend=None, ctx=None)  # type: ignore[arg-type]
    assert len(out) == 1
    assert clone.seen[-1] == "Zeta."  # not "TwZeta."


# ---------------------------------------------------------------------------
# Delta-invariance through the requirement layer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("seed", range(5))
async def test_delta_invariance_through_requirement(seed):
    """Any delta slicing of the same text yields the same units as one split + flush."""
    full = "Alpha one. Beta two! Gamma three? Delta four"

    reference = Chunker(SentenceChunking())
    expected = reference.feed(full) + reference.flush()

    rng = random.Random(seed)
    # Slice `full` into random deltas.
    cuts = sorted(rng.sample(range(1, len(full)), rng.randint(0, len(full) - 1)))
    bounds = [0, *cuts, len(full)]
    deltas = [full[a:b] for a, b in pairwise(bounds) if a < b]

    req = RecordingReq(chunking="sentence")
    for d in deltas:
        await req.stream_validate(d, backend=None, ctx=None)  # type: ignore[arg-type]
    await req.stream_flush(backend=None, ctx=None)  # type: ignore[arg-type]

    assert req.seen == expected


# ---------------------------------------------------------------------------
# PartialValidationSummary.from_results
# ---------------------------------------------------------------------------


def test_summary_all_pass():
    results = [PartialValidationResult("pass"), PartialValidationResult("pass")]
    summary = PartialValidationSummary.from_results(results)
    assert summary.success == "pass"
    assert summary.failure is None
    assert summary.reason is None
    assert summary.results == results


def test_summary_any_fail():
    fail = PartialValidationResult("fail", reason="nope")
    summary = PartialValidationSummary.from_results(
        [PartialValidationResult("pass"), fail, PartialValidationResult("unknown")]
    )
    assert summary.success == "fail"
    assert summary.failure is fail
    assert summary.reason == "nope"


def test_summary_first_failure_wins():
    first = PartialValidationResult("fail", reason="first")
    second = PartialValidationResult("fail", reason="second")
    summary = PartialValidationSummary.from_results([first, second])
    assert summary.failure is first
    assert summary.reason == "first"


def test_summary_mixed_without_fail_is_unknown():
    summary = PartialValidationSummary.from_results(
        [PartialValidationResult("pass"), PartialValidationResult("unknown")]
    )
    assert summary.success == "unknown"
    assert summary.failure is None
    assert summary.reason is None


def test_summary_empty_is_unknown():
    summary = PartialValidationSummary.from_results([])
    assert summary.success == "unknown"
    assert summary.failure is None
    assert summary.reason is None
