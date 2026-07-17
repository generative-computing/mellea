# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import TypedDict


class ICLExample(TypedDict):
    execution_plan: list[str]
    available_content_variables: list[str]
    target_subtask: str
    subtask_prompt_template: str


class ICLExampleGroup(TypedDict):
    task_prompt: str
    examples_items: list[ICLExample]
