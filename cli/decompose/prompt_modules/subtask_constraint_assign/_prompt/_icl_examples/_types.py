# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import TypedDict


class ICLExample(TypedDict):
    execution_plan: list[str]
    constraint_list: list[str]
    subtask_title: str
    subtask_prompt: str
    assigned_constraints: list[str]
