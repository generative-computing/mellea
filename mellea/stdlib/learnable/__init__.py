# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Learning functions into adapters.

* `learnable`: `@learnable` functions, their learnable view and output type,
  the policy that answers a call (`Learnable.policy`), and the `Record` each
  call leaves inside a `Recorder` (a tree: each record holds the calls its body
  made; `keep` and `every` record only some).
* `capture`: a driver run over episodes, its calls scored and written
  (`read_rows` reads them back); `sft_rows` and `preference_rows`, the
  completions as training rows.
* `instantiate`: the functions of a tree learned inner first, each pass
  captured again with the students trained so far installed.
"""

from .capture import (
    Scorer,
    capture,
    preference_rows,
    read_rows,
    sft_rows,
    walk,
    walk_rows,
)
from .instantiate import Trainer, instantiate, levels
from .learnable import (
    VIEW,
    Keep,
    Learnable,
    Policy,
    Record,
    Recorder,
    Returned,
    Score,
    Student,
    every,
    learnable,
    learnable_view,
)

__all__ = [
    "VIEW",
    "Keep",
    "Learnable",
    "Policy",
    "Record",
    "Recorder",
    "Returned",
    "Score",
    "Scorer",
    "Student",
    "Trainer",
    "capture",
    "every",
    "instantiate",
    "learnable",
    "learnable_view",
    "levels",
    "preference_rows",
    "read_rows",
    "sft_rows",
    "walk",
    "walk_rows",
]
