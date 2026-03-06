# SPDX-License-Identifier: Apache-2.0

"""Input and output processing code for Granite models and for Granite intrinsics."""

# Local
# This file explicitly imports all the symbols that we export at the top level of this
# package's namespace.
from .base.types import (
    AssistantMessage,
    ChatCompletion,
    ChatCompletionResponse,
    DocumentMessage,
    GraniteChatCompletion,
    UserMessage,
    VLLMExtraBody,
)
from .granite3.granite32 import (
    Granite32ChatCompletion,
    Granite32InputProcessor,
    Granite32OutputProcessor,
)
from .granite3.granite33 import (
    Granite33ChatCompletion,
    Granite33InputProcessor,
    Granite33OutputProcessor,
)
from .intrinsics import IntrinsicsResultProcessor, IntrinsicsRewriter

__all__ = [
    "AssistantMessage",
    "ChatCompletion",
    "ChatCompletionResponse",
    "DocumentMessage",
    "Granite32ChatCompletion",
    "Granite32InputProcessor",
    "Granite32OutputProcessor",
    "Granite33ChatCompletion",
    "Granite33InputProcessor",
    "Granite33OutputProcessor",
    "GraniteChatCompletion",
    "IntrinsicsResultProcessor",
    "IntrinsicsRewriter",
    "UserMessage",
    "VLLMExtraBody",
]
