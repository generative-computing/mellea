# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import TypedDict


class ICLExample(TypedDict):
    task_prompt: str
    thinking_process: str
    subtask_list: list[str]
