# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from ._example_1 import example as example_1
from ._example_2 import example as example_2
from ._types import ICLExample

icl_examples: list[ICLExample] = [example_1, example_2]
