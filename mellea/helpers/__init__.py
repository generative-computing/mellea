"""Low-level helpers and utilities supporting mellea backends.

This package provides the internal plumbing used by the built-in backend
implementations: async utilities (`send_to_queue`, `wait_for_all_mots`,
`ClientCache`) for managing concurrent model-output thunks; OpenAI-compatible
message conversion helpers (`message_to_openai_message`, `messages_to_docs`,
`chat_completion_delta_merge`); and `_ServerType` detection for adapting
structured-output support to the target server. Most user code will not import from
this package directly — it is consumed internally by the backend layer.
"""

from .async_helpers import (
    ClientCache,
    get_current_event_loop,
    send_to_queue,
    wait_for_all_mots,
)
from .event_loop_helper import _run_async_in_thread
from .openai_compatible_helpers import (
    chat_completion_delta_merge,
    extract_model_tool_requests,
    message_to_openai_message,
    messages_to_docs,
)
from .server_type import (
    _server_type,
    _ServerType,
    is_vllm_server_with_structured_output,
)

__all__ = [
    "ClientCache",
    "_ServerType",
    "_run_async_in_thread",
    "_server_type",
    "chat_completion_delta_merge",
    "extract_model_tool_requests",
    "get_current_event_loop",
    "is_vllm_server_with_structured_output",
    "message_to_openai_message",
    "messages_to_docs",
    "send_to_queue",
    "wait_for_all_mots",
]
