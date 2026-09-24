# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The budget-forcing retry must receive the context returned by repair."""

from types import SimpleNamespace

import pytest

from mellea.backends import ollama
from mellea.core import CBlock
from mellea.stdlib.sampling import budget_forcing


@pytest.mark.asyncio
async def test_retry_uses_repaired_context(monkeypatch):
    class FakeOllamaBackend:
        pass

    class FakeThunk:
        def __init__(self):
            self.value = "answer"
            self._generate_log = SimpleNamespace(is_final_result=False)

        async def avalue(self):
            return self.value

    original_context = object()
    repaired_context = object()
    seen_contexts = []
    validation_count = 0

    async def fake_generate(*args, ctx, **kwargs):
        seen_contexts.append(ctx)
        return FakeThunk()

    async def fake_validate(**kwargs):
        nonlocal validation_count
        validation_count += 1
        return [validation_count == 2]

    monkeypatch.setattr(ollama, "OllamaModelBackend", FakeOllamaBackend)
    monkeypatch.setattr(budget_forcing, "think_budget_forcing", fake_generate)
    monkeypatch.setattr(budget_forcing, "ComputedModelOutputThunk", lambda thunk: thunk)
    monkeypatch.setattr(budget_forcing.mfuncs, "avalidate", fake_validate)
    strategy = budget_forcing.BudgetForcingSamplingStrategy(
        loop_budget=2, requirements=[object()]
    )
    monkeypatch.setattr(
        strategy, "repair", lambda *args: (CBlock("repair"), repaired_context)
    )

    result = await strategy.sample(
        CBlock("first attempt"),
        original_context,
        FakeOllamaBackend(),
        None,
        show_progress=False,
    )

    assert result.success is True
    assert seen_contexts == [original_context, repaired_context]
