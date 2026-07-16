# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import TypedDict


class ICLExample(TypedDict):
    requirement: str
    reasoning: str
    decision: str
