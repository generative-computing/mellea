# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Learning functions into adapters.

* ``learnable``: ``@learnable`` functions (each an ``AdapterFunction`` whose
  teacher is its body), the policy that answers a call (``Learnable.policy``;
  ``adapter_student`` serves the trained adapter), and the ``Record`` each call
  leaves inside a ``Recorder`` (a tree: each record holds the calls its body
  made, and the call as the student backend renders it; ``keep`` and
  ``every`` record only some).
* ``capture``: a driver run over episodes, its calls scored and written
  (``read_rows`` reads them back); ``sft_rows`` and ``preference_rows``, the
  thunks as training rows.
* ``instantiate``: the functions of a tree learned inner first, each pass
  captured again with the students trained so far installed.
"""

from ..components.intrinsic.adapter_function import AdapterFunction, OutputSpec
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
    AdapterStudent,
    Keep,
    Learnable,
    Policy,
    Record,
    Recorder,
    Returned,
    Score,
    Student,
    Thunk,
    adapter_student,
    every,
    learnable,
)

__all__ = [
    "AdapterFunction",
    "AdapterStudent",
    "Keep",
    "Learnable",
    "OutputSpec",
    "Policy",
    "Record",
    "Recorder",
    "Returned",
    "Score",
    "Scorer",
    "Student",
    "Thunk",
    "Trainer",
    "adapter_student",
    "capture",
    "every",
    "instantiate",
    "learnable",
    "levels",
    "preference_rows",
    "read_rows",
    "sft_rows",
    "walk",
    "walk_rows",
]
