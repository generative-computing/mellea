# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from ._example_group_1 import example_group as example_group_1
from ._example_group_2 import example_group as example_group_2
from ._types import ICLExampleGroup

# icl_example_groups: list[ICLExampleGroup] = [example_group_1]
# icl_example_groups: list[ICLExampleGroup] = [example_group_2]
icl_example_groups: list[ICLExampleGroup] = [example_group_1, example_group_2]
