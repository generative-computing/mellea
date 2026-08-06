---
id: cache
title: "mellea.backends.cache"
sidebar_label: "cache"
sidebar_position: 4
description: "Cache abstractions and implementations for model state."
# diataxis: reference
---

Source: [`mellea/backends/cache.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/cache.py) at commit `a535fc6345a0`.

Cache abstractions and implementations for model state.

## `Cache`

*class* — [line 19](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/cache.py#L19) (`abc.ABC`)

A Cache for storing model state (e.g., kv cache).

Methods (defined on this class; inherited members not listed):

- `put(key: str | int, value: Any) -> None` — [line 25](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/cache.py#L25)
  Insert a value into the cache under the given key.
- `get(key: str | int) -> Any | None` — [line 37](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/cache.py#L37)
  Retrieve a value from the cache by key.
- `current_size() -> int` — [line 51](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/cache.py#L51)
  Return the number of entries currently stored in the cache.

## `SimpleLRUCache`

*class* — [line 60](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/cache.py#L60) (`Cache`)

A simple `LRU <https://en.wikipedia.org/wiki/Cache_replacement_policies#Least_Recently_Used_(LRU)>`_ cache.

Constructor: `SimpleLRUCache(capacity: int, on_evict: Callable[[Any], None] | None = None)`

Methods (defined on this class; inherited members not listed):

- `current_size() -> int` — [line 83](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/cache.py#L83)
  Return the number of entries currently stored in the cache.
- `get(key: str | int) -> Any | None` — [line 91](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/cache.py#L91)
  Retrieve a value from the cache, promoting it to most-recently-used.
- `put(key: str | int, value: Any) -> None` — [line 108](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/cache.py#L108)
  Insert or update a value in the cache.

---

*Generated from source by `docs/scripts/generate_package_map.py` — do not edit by hand; regenerate instead. Symbols and annotations reflect the pinned commit exactly; where type annotations are absent in source, this page says so rather than guessing.*
