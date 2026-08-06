---
id: server_type
title: "mellea.helpers.server_type"
sidebar_label: "server_type"
sidebar_position: 4
description: "Utilities for detecting and classifying the target inference server."
# diataxis: reference
---

Source: [`mellea/helpers/server_type.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/helpers/server_type.py) at commit `a535fc6345a0`.

Utilities for detecting and classifying the target inference server.

## `is_vllm_server_with_structured_output()`

*function* — [line 89](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/helpers/server_type.py#L89)

`is_vllm_server_with_structured_output(base_url: str, headers: Mapping[str, Any]) -> bool`

Attempts to determine if the backend is a vllm server with version >= v0.12.0. Defaults to false.

---

*Generated from source by `docs/scripts/generate_package_map.py` — do not edit by hand; regenerate instead. Symbols and annotations reflect the pinned commit exactly; where type annotations are absent in source, this page says so rather than guessing.*
