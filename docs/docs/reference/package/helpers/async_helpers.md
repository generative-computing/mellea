---
id: async_helpers
title: "mellea.helpers.async_helpers"
sidebar_label: "async_helpers"
sidebar_position: 1
description: "Async helper functions for managing concurrent model output thunks."
# diataxis: reference
---

Source: [`mellea/helpers/async_helpers.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/helpers/async_helpers.py) at commit `a535fc6345a0`.

Async helper functions for managing concurrent model output thunks.

## `ClientCache`

*class* — [line 164](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/helpers/async_helpers.py#L164) 

A simple [LRU](https://en.wikipedia.org/wiki/Cache_replacement_policies#Least_Recently_Used_(LRU)) cache.

Constructor: `ClientCache(capacity: int)`

Methods (defined on this class; inherited members not listed):

- `current_size() -> int` — [line 182](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/helpers/async_helpers.py#L182)
  Just return the size of the key set. This isn't necessarily safe.
- `get(key: int) -> Any | None` — [line 190](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/helpers/async_helpers.py#L190)
  Gets a value from the cache.
- `put(key: int, value: Any) -> None` — [line 207](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/helpers/async_helpers.py#L207)
  Put a value into the cache.

## `send_to_queue()`

*async function* — [line 39](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/helpers/async_helpers.py#L39)

`send_to_queue(co: Coroutine[Any, Any, AsyncIterator | Any] | AsyncIterator, aqueue: asyncio.Queue, *, chunk_timeout: float | None = DEFAULT_CHUNK_TIMEOUT, on_timeout: Callable[[], None] | None = None) -> None`

Processes the output of an async chat request by sending the output to an async queue.

## `wait_for_all_mots()`

*async function* — [line 134](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/helpers/async_helpers.py#L134)

`wait_for_all_mots(mots: list[ModelOutputThunk]) -> None`

Helper function to make waiting for multiple ModelOutputThunks to be computed easier.

## `get_current_event_loop()`

*function* — [line 150](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/helpers/async_helpers.py#L150)

`get_current_event_loop() -> None | asyncio.AbstractEventLoop`

Get the current event loop without having to catch exceptions.

---

*Generated from source by `docs/scripts/generate_package_map.py` — do not edit by hand; regenerate instead. Symbols and annotations reflect the pinned commit exactly; where type annotations are absent in source, this page says so rather than guessing.*
