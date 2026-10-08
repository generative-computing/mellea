# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the two automatic LLMON core seams.

LLMON behavior is driven by the served model id (`llmon_models.is_llmon_model`) at
two seams:

- Outbound-from-model strip: `ModelOutputThunk._set_parsed_repr` strips Machine
  LLMON out of the reply into a plain-`str` `parsed_repr` for an LLMON model.
- Inbound-to-model encode: `Backend.generate_from_context` wraps an `Instruction`
  so it renders as Machine LLMON, and rejects a Pydantic `format=`.

These tests exercise the gates directly, without a served endpoint.
"""

import pytest

# LLMON support is an optional extra; the strip seam imports `parse_llmon_reply`
# from the `llmon` library once a model matches, so skip without it.
pytest.importorskip("lang.json.llmon_converters")

from lang.common import common as _c

from mellea.core.base import ModelOutputThunk
from mellea.stdlib.components.instruction import Instruction

_ASSISTANT_BLOCK = (
    f"{_c.OPEN_TAG}assistant{_c.COLON_SEPARATOR_TAG}1{_c.CLOSE_TAG}"
    f"the answer{_c.OPEN_END_TAG}"
)


def _computed_thunk(value: str, model: str) -> ModelOutputThunk:
    """Build a minimally-computed thunk with a model id and an Instruction action."""
    t = ModelOutputThunk(value=None)
    t._underlying_value = value
    t._computed = True
    t.generation.model = model
    t._call.action = Instruction("Task.")
    return t


class TestStripSeam:
    """`_set_parsed_repr` strips LLMON to a plain `str` only for an LLMON model."""

    def test_llmon_model_strips_to_plain_str(self):
        t = _computed_thunk(_ASSISTANT_BLOCK, "llmon95:latest")
        t._set_parsed_repr()
        assert t.parsed_repr == "the answer"
        assert isinstance(t.parsed_repr, str)

    def test_raw_value_is_preserved(self):
        # Only parsed_repr holds the stripped text; value stays the raw record.
        t = _computed_thunk(_ASSISTANT_BLOCK, "llmon95:latest")
        t._set_parsed_repr()
        assert t.value == _ASSISTANT_BLOCK
        assert t.value != t.parsed_repr

    def test_plain_reply_from_llmon_model_passes_through(self):
        t = _computed_thunk("already plain", "llmon95:latest")
        t._set_parsed_repr()
        assert t.parsed_repr == "already plain"

    def test_non_llmon_model_uses_normal_parse(self):
        # Not allowlisted: the Instruction._parse path runs, returning the raw value.
        t = _computed_thunk(_ASSISTANT_BLOCK, "granite4.2:3b")
        t._set_parsed_repr()
        assert t.parsed_repr == _ASSISTANT_BLOCK


def _make_stub_backend(model_id: str):
    """Build a Backend stub whose `_generate_from_context` records the action."""
    import asyncio

    from mellea.core.backend import Backend
    from mellea.core.base import ModelOutputThunk

    class _Stub(Backend):
        _model_id = model_id
        _provider = "stub"

        def __init__(self):
            self.seen_action = None

        async def _generate_from_context(
            self, action, ctx, *, format=None, model_options=None, tool_calls=False
        ):
            self.seen_action = action
            out = ModelOutputThunk(value=None)
            out._underlying_value = "ok"
            out._computed = True
            return out, ctx

        async def _generate_from_raw(self, *args, **kwargs):  # pragma: no cover
            raise NotImplementedError

    return _Stub(), asyncio


class TestEncodeSeam:
    """`Backend.generate_from_context` encode gate: wrap Instruction, reject format=.

    Uses a stub backend so the real concrete `generate_from_context` runs its LLMON
    gate with no network; a full round-trip is the e2e example's job.
    """

    def test_llmon_model_wraps_instruction_as_machine_llmon(self):
        from mellea.formatters.template_formatter import TemplateFormatter
        from mellea.stdlib.components.llmon import is_llmon

        backend, asyncio = _make_stub_backend("llmon95:latest")
        asyncio.run(backend.generate_from_context(Instruction("Summarize this."), None))
        # The action passed downstream renders as Machine LLMON, not prose.
        rendered = TemplateFormatter(model_id="default").print(backend.seen_action)
        assert is_llmon(rendered)

    def test_non_llmon_model_passes_instruction_through_unchanged(self):
        backend, asyncio = _make_stub_backend("granite4.2:3b")
        action = Instruction("Summarize this.")
        asyncio.run(backend.generate_from_context(action, None))
        # Identity: no wrapping for a non-allowlisted model.
        assert backend.seen_action is action

    def test_llmon_model_rejects_format(self):
        import pydantic

        class _Schema(pydantic.BaseModel):
            x: int

        backend, asyncio = _make_stub_backend("llmon95:latest")
        with pytest.raises(ValueError, match="format="):
            asyncio.run(
                backend.generate_from_context(
                    Instruction("Task."), None, format=_Schema
                )
            )


def _sampling_backend(model_id: str, raw_reply: str, parsed_repr: str):
    """A Backend whose generation returns a pre-computed thunk, as a real backend
    leaves it after `_set_parsed_repr`: raw LLMON in `value`, stripped text in
    `parsed_repr`. Lets the real sampling loop run with no network."""
    import asyncio

    from mellea.core.backend import Backend
    from mellea.core.base import ModelOutputThunk

    class _Stub(Backend):
        _model_id = model_id
        _provider = "stub"

        async def _generate_from_context(
            self, action, ctx, *, format=None, model_options=None, tool_calls=False
        ):
            from mellea.core.base import GenerateLog

            out = ModelOutputThunk(value=None)
            out._underlying_value = raw_reply
            out._computed = True
            out.generation.model = model_id
            out._call.action = action
            out.parsed_repr = parsed_repr  # as _set_parsed_repr left it
            glog = GenerateLog()
            glog.action = action
            out._generate_log = glog
            return out, ctx

        async def _generate_from_raw(self, *args, **kwargs):  # pragma: no cover
            raise NotImplementedError

    return _Stub(), asyncio


class TestSamplingPreservesStrip:
    """A sampling strategy must not clobber the LLMON-stripped parsed_repr.

    Regression: the subsample loop in `sampling/base.py` (and `budget_forcing.py`)
    re-parsed `result.parsed_repr = action.parse(result)` after generation, which
    re-ran the base `Instruction._parse` and overwrote the model-driven strip that
    `_set_parsed_repr` set — handing the judge (and the caller) raw Machine LLMON
    for an LLMON model. The re-parse is now gated off for an LLMON model.
    """

    def test_rejection_sampling_keeps_stripped_parsed_repr(self):
        from mellea.stdlib.context import ChatContext
        from mellea.stdlib.sampling.base import RejectionSamplingStrategy

        backend, asyncio = _sampling_backend(
            "llmon95:latest", raw_reply=_ASSISTANT_BLOCK, parsed_repr="the answer"
        )
        strategy = RejectionSamplingStrategy(loop_budget=1)

        async def _run():
            return await strategy.sample(
                Instruction("Task."),
                context=ChatContext(),
                backend=backend,
                requirements=[],
            )

        sampling_result = asyncio.run(_run())
        # The chosen result's parsed_repr must still be the stripped plain text, not
        # the raw Machine LLMON re-parsed from the action.
        assert str(sampling_result.result.parsed_repr) == "the answer"
        assert "<|open|>" not in str(sampling_result.result.parsed_repr)
