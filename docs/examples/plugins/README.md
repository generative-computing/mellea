# Plugin Examples

Examples demonstrating how to extend Mellea with hooks and plugins for observability,
modification, and control over the generation pipeline. Plugins need the `hooks`
extra (`uv sync --extra hooks`), and all but `testing_plugins.py` run against a local
Ollama model. For the concepts, the full hook reference, and execution-mode semantics,
see [Plugins](../../docs/concepts/plugins.mdx). For the built-in debug plugins, see
[Debug with plugins](../../docs/how-to/debug-with-plugins.md).

## Examples

| File | Description |
| :--- | :--- |
| [`quickstart.py`](./quickstart.py) | Registering one function hook on `GENERATION_PRE_CALL` to log every generation call. |
| [`standalone_hooks.py`](./standalone_hooks.py) | Function-based hooks that enforce a token budget and log generation latency. |
| [`class_plugin.py`](./class_plugin.py) | Grouping hooks in a `Plugin` subclass for PII protection: block SSNs on input, scan output. |
| [`execution_modes.py`](./execution_modes.py) | All five `PluginMode` values (`SEQUENTIAL`, `TRANSFORM`, `AUDIT`, `CONCURRENT`, `FIRE_AND_FORGET`) on one hook type. |
| [`payload_modification.py`](./payload_modification.py) | Changing payloads with `modify()` and `model_copy(update=...)`, and what happens to non-writable fields. |
| [`plugin_set_composition.py`](./plugin_set_composition.py) | Grouping hooks into `PluginSet`s by concern, registered globally or per session. |
| [`session_scoped.py`](./session_scoped.py) | Global hooks versus plugins passed via `start_session(plugins=...)` that fire for one session only. |
| [`plugin_scoped.py`](./plugin_scoped.py) | Block-scoped activation with `plugin_scope()`, `Plugin` and `PluginSet` as context managers. |
| [`tool_hooks.py`](./tool_hooks.py) | `TOOL_PRE_INVOKE` / `TOOL_POST_INVOKE` hooks for tool allow-listing, argument validation, audit logging, and argument sanitizing. |
| [`testing_plugins.py`](./testing_plugins.py) | Unit-testing hook functions with hand-built payloads and no live session. |
| [`builtin_generation_tracing.py`](./builtin_generation_tracing.py) | Built-in generation tracing: model, prompt and response previews, latency, and token usage. |
| [`builtin_validation_tracing.py`](./builtin_validation_tracing.py) | Built-in validation tracing: per-requirement pass/fail with reasons. |
| [`builtin_validation_failures.py`](./builtin_validation_failures.py) | Validation tracing with immediate validation and no repair, so real failures are logged. |
| [`builtin_validation_strict.py`](./builtin_validation_strict.py) | Validation tracing with strict requirements designed to fail. |
| [`builtin_sampling_diagnostics.py`](./builtin_sampling_diagnostics.py) | Built-in sampling tracing: loop budget, per-iteration results, repair events, and final outcome. |
| [`builtin_full_pipeline_tracing.py`](./builtin_full_pipeline_tracing.py) | Generation and sampling tracing together across the whole sampling loop. |
| [`builtin_complete_diagnostics.py`](./builtin_complete_diagnostics.py) | Generation, sampling, and validation tracing combined for end-to-end visibility. |
