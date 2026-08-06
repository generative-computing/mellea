---
id: template_formatter
title: "mellea.formatters.template_formatter"
sidebar_label: "template_formatter"
sidebar_position: 3
description: "`TemplateFormatter`: Jinja2-template-based formatter for legacy backends."
# diataxis: reference
---

Source: [`mellea/formatters/template_formatter.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/template_formatter.py) at commit `a535fc6345a0`.

`TemplateFormatter`: Jinja2-template-based formatter for legacy backends.

## `TemplateFormatter`

*class* — [line 38](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/template_formatter.py#L38) (`ChatFormatter`)

Formatter that uses Jinja2 templates to render components into prompt strings.

Constructor: `TemplateFormatter(model_id: str | ModelIdentifier, *, template_path: str = '', use_template_cache: bool = True)`

Methods (defined on this class; inherited members not listed):

- `print(c: Span) -> str` — [line 171](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/template_formatter.py#L171)
  Render a component, content block, or model output to a string using a Jinja2 template.

---

*Generated from source by `docs/scripts/generate_package_map.py` — do not edit by hand; regenerate instead. Symbols and annotations reflect the pinned commit exactly; where type annotations are absent in source, this page says so rather than guessing.*
