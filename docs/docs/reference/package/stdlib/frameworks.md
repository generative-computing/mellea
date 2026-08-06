---
id: frameworks
title: "mellea.stdlib.frameworks"
sidebar_label: "frameworks"
sidebar_position: 4
description: "Problem solving frameworks."
# diataxis: reference
---

Source: [`mellea/stdlib/frameworks/__init__.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/frameworks/__init__.py) at commit `a535fc6345a0`.

Problem solving frameworks.

---

## Module `mellea.stdlib.frameworks.react`

Source: [`mellea/stdlib/frameworks/react.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/frameworks/react.py) at commit `a535fc6345a0`.

ReACT (Reason + Act) agentic pattern implementation.

### `react()`

*async function* — [line 30](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/frameworks/react.py#L30)

`react(goal: str, context: ChatContext, backend: Backend, *, format: type[BaseModelSubclass] | None = None, model_options: dict | None = None, tools: list[AbstractMelleaTool] | None, loop_budget: int = 10, compactor: Compactor | None = None) -> tuple[ComputedModelOutputThunk[str], ChatContext]`

Asynchronous ReACT pattern (Think -> Act -> Observe -> Repeat Until Done); attempts to accomplish the provided goal given the provided tools.

---

*Generated from source by `docs/scripts/generate_package_map.py` — do not edit by hand; regenerate instead. Symbols and annotations reflect the pinned commit exactly; where type annotations are absent in source, this page says so rather than guessing.*
