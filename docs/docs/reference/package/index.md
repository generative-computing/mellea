---
id: index
title: "Package map"
sidebar_label: "Overview"
sidebar_position: 0
description: "A source-pinned, machine-generated map of mellea's public import surface: every public module, class, and function linked to its source line."
# diataxis: reference
---

A machine-generated map of `mellea`'s **public import surface**, pinned to commit
[`a535fc6345a0`](https://github.com/generative-computing/mellea/commit/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5):
**140 public modules** exposing **246 public classes**
and **218 public module-level functions**, every one linked to its
source line. Where the source lacks type annotations, pages say so explicitly rather than
inventing types (see issue
[#1177](https://github.com/generative-computing/mellea/issues/1177)).

Counting method: a module is public when no component of its dotted path starts with `_`;
a symbol is public when it is a top-level `class`/`def` without a leading underscore.
Everything below is derived from the AST of the pinned source — no imports, no inference.

## Subpackages

- [`mellea.backends`](backends/index.md) — Backend implementations for the mellea inference layer. *(20 public modules)*
- [`mellea.core`](core/index.md) — Core abstractions for the mellea library. *(7 public modules)*
- [`mellea.formatters`](formatters/index.md) — Formatters for converting components into model-ready prompts. *(19 public modules)*
- [`mellea.helpers`](helpers/index.md) — Low-level helpers and utilities supporting mellea backends. *(5 public modules)*
- [`mellea.plugins`](plugins/index.md) — Mellea Plugin System — extension points for policy enforcement, observability, and customization. *(22 public modules)*
- [`mellea.serve`](serve/index.md) — Public API for m serve types. *(2 public modules)*
- [`mellea.stdlib`](stdlib/index.md) — The mellea standard library of components, sessions, and sampling strategies. *(56 public modules)*
- [`mellea.telemetry`](telemetry/index.md) — OpenTelemetry instrumentation for Mellea. *(8 public modules)*

## Root exports

`mellea/__init__.py` declares `__all__` = `MelleaSession`, `generative`, `model_ids`, `serve`, `start_backend`, `start_session` ([source](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/__init__.py)).

---

*Generated from source by `docs/scripts/generate_package_map.py` — do not edit by hand; regenerate instead. Symbols and annotations reflect the pinned commit exactly; where type annotations are absent in source, this page says so rather than guessing.*
