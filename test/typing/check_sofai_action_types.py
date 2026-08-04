# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mypy checks that SOFAI's S2 helpers accept the widened action union.

After #1439, `SOFAISamplingStrategy._extract_action_prompt`,
`_prepare_s2_context`, and `_generate_and_validate` accept `Span`
(`Component | CBlock | ModelOutputThunk`) rather than `Component` only. A bare
`CBlock` or `ModelOutputThunk` action that escalates to the S2 "best_attempt"
path must type-check without a `# type: ignore[arg-type]` suppression. These
functions are never executed; `uv run mypy .` verifies them.
"""

from typing import cast

from mellea.core import Backend, CBlock, Context, ModelOutputThunk, Requirement
from mellea.core.base import ComputedModelOutputThunk

# Runtime import; used only for its type in the check below.
from mellea.stdlib.sampling.sofai import SOFAISamplingStrategy

cblock_action: CBlock = cast(CBlock, None)
mot_action: ModelOutputThunk[str] = cast(ModelOutputThunk[str], None)

# Placeholder collaborators for the multi-arg helpers below. These functions are
# never executed, so `None` cast to the expected type is sufficient for mypy.
strategy: SOFAISamplingStrategy = cast(SOFAISamplingStrategy, None)
ctx: Context = cast(Context, None)
backend: Backend = cast(Backend, None)
reqs: list[Requirement] = cast("list[Requirement]", None)
sampled_results: list[ComputedModelOutputThunk] = cast(
    "list[ComputedModelOutputThunk]", None
)


def check_extract_action_prompt_accepts_cblock() -> None:
    # Would raise arg-type under the pre-#1439 Component-only signature.
    SOFAISamplingStrategy._extract_action_prompt(cblock_action)


def check_extract_action_prompt_accepts_mot() -> None:
    SOFAISamplingStrategy._extract_action_prompt(mot_action)


def check_prepare_s2_context_accepts_cblock_and_mot() -> None:
    # Both the `original_action` and `last_action` parameters take the widened
    # union; a bare CBlock or ModelOutputThunk must not raise arg-type.
    strategy._prepare_s2_context(
        "best_attempt", cblock_action, ctx, ctx, mot_action, sampled_results, [], 0
    )


async def check_generate_and_validate_accepts_cblock() -> None:
    # The `action` parameter takes the widened union.
    await strategy._generate_and_validate(
        backend, cblock_action, ctx, reqs, backend, None, None, False
    )


async def check_generate_and_validate_accepts_mot() -> None:
    await strategy._generate_and_validate(
        backend, mot_action, ctx, reqs, backend, None, None, False
    )
