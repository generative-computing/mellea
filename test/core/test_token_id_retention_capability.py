# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A retention policy the bound backend cannot honour must warn, not pass silently."""

import pytest

from mellea.backends.openai import OpenAIBackend
from mellea.core.backend import Backend
from mellea.stdlib.context import ChatContext


def test_openai_backend_declares_support() -> None:
    """The backend that sends retained ids advertises it, so no warning fires for it."""
    assert OpenAIBackend._supports_token_id_retention is True


def test_backends_do_not_declare_support_by_default() -> None:
    """Every other backend re-renders from text, so the base default must stay off."""
    assert Backend._supports_token_id_retention is False


@pytest.mark.asyncio
async def test_unsupported_backend_warns_once(caplog) -> None:
    """A retaining context on a non-supporting backend warns, once per instance."""

    class _Unsupporting(Backend):
        _model_id = "fake/model"
        _provider = "fake"

        async def _generate_from_context(self, action, ctx, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("not reached: the warning fires before dispatch")

        async def _generate_from_raw(self, actions, ctx, **kwargs):  # type: ignore[no-untyped-def]
            raise NotImplementedError

    backend = _Unsupporting()
    ctx = ChatContext(retain_token_ids=True)

    for _ in range(2):
        with pytest.raises(RuntimeError):
            await backend.generate_from_context("hi", ctx)  # type: ignore[arg-type]

    warnings = [
        r for r in caplog.records if "asked to retain token ids" in r.getMessage()
    ]
    assert len(warnings) == 1, "the warning must fire once per instance, not per turn"
    assert "_Unsupporting" in warnings[0].getMessage()


@pytest.mark.asyncio
async def test_non_retaining_context_never_warns(caplog) -> None:
    """A plain context must not warn, whatever the backend supports."""

    class _Unsupporting(Backend):
        _model_id = "fake/model"
        _provider = "fake"

        async def _generate_from_context(self, action, ctx, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("not reached")

        async def _generate_from_raw(self, actions, ctx, **kwargs):  # type: ignore[no-untyped-def]
            raise NotImplementedError

    with pytest.raises(RuntimeError):
        await _Unsupporting().generate_from_context("hi", ChatContext())  # type: ignore[arg-type]

    assert not [
        r for r in caplog.records if "asked to retain token ids" in r.getMessage()
    ]
