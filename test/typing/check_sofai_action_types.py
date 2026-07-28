# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mypy checks that SOFAI's S2 helpers accept the widened action union.

After #1439, `SOFAISamplingStrategy._extract_action_prompt`,
`_prepare_s2_context`, and `_generate_and_validate` accept `NodeData`
(`Component | CBlock | ModelOutputThunk`) rather than `Component` only. A bare
`CBlock` or `ModelOutputThunk` action that escalates to the S2 "best_attempt"
path must type-check without a `# type: ignore[arg-type]` suppression. These
functions are never executed; `uv run mypy .` verifies them.
"""

from typing import cast

from mellea.core import CBlock, ModelOutputThunk

# Runtime import; used only for its type in the check below.
from mellea.stdlib.sampling.sofai import SOFAISamplingStrategy

cblock_action: CBlock = cast(CBlock, None)
mot_action: ModelOutputThunk[str] = cast(ModelOutputThunk[str], None)


def check_extract_action_prompt_accepts_cblock() -> None:
    # Would raise arg-type under the pre-#1439 Component-only signature.
    SOFAISamplingStrategy._extract_action_prompt(cblock_action)


def check_extract_action_prompt_accepts_mot() -> None:
    SOFAISamplingStrategy._extract_action_prompt(mot_action)
