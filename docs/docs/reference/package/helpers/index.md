---
id: index
title: "mellea.helpers"
sidebar_label: "Overview"
sidebar_position: 0
description: "Low-level helpers and utilities supporting mellea backends."
# diataxis: reference
---

Source: [`mellea/helpers/__init__.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/helpers/__init__.py) at commit `a535fc6345a0`.

Low-level helpers and utilities supporting mellea backends.

Declared exports (`__all__`): `DEFAULT_CHUNK_TIMEOUT`, `ClientCache`, `_ServerType`, `_run_async_in_thread`, `_server_type`, `chat_completion_delta_merge`, `extract_model_tool_requests`, `get_current_event_loop`, `is_vllm_server_with_structured_output`, `message_to_openai_message`, `messages_to_docs`, `send_to_queue`, `should_replay_reasoning`, `wait_for_all_mots`

## Modules

- [`mellea.helpers.async_helpers`](async_helpers.md) — Async helper functions for managing concurrent model output thunks.
- [`mellea.helpers.event_loop_helper`](event_loop_helper.md) — Helper for event loop management. Allows consistently running async generate requests in sync code.
- [`mellea.helpers.openai_compatible_helpers`](openai_compatible_helpers.md) — A file for helper functions that deal with OpenAI API compatible helpers.
- [`mellea.helpers.server_type`](server_type.md) — Utilities for detecting and classifying the target inference server.

---

*Generated from source by `docs/scripts/generate_package_map.py` — do not edit by hand; regenerate instead. Symbols and annotations reflect the pinned commit exactly; where type annotations are absent in source, this page says so rather than guessing.*
