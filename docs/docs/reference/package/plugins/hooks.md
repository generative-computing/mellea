---
id: hooks
title: "mellea.plugins.hooks"
sidebar_label: "hooks"
sidebar_position: 5
description: "Hook payload classes for the Mellea plugin system."
# diataxis: reference
---

Source: [`mellea/plugins/hooks/__init__.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/hooks/__init__.py) at commit `a535fc6345a0`.

Hook payload classes for the Mellea plugin system.

Declared exports (`__all__`): `AdapterFunctionInvocationCompletePayload`, `AdapterFunctionPhaseCompletePayload`, `ComponentPostErrorPayload`, `ComponentPostSuccessPayload`, `ComponentPreExecutePayload`, `GenerationPostCallPayload`, `GenerationPreCallPayload`, `SamplingIterationPayload`, `SamplingLoopEndPayload`, `SamplingLoopStartPayload`, `SamplingRepairPayload`, `SessionCleanupPayload`, `SessionPostInitPayload`, `SessionPreInitPayload`, `SessionResetPayload`, `ToolPostInvokePayload`, `ToolPreInvokePayload`, `ValidationPostCheckPayload`, `ValidationPreCheckPayload`

---

## Module `mellea.plugins.hooks.adapter_function`

Source: [`mellea/plugins/hooks/adapter_function.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/hooks/adapter_function.py) at commit `a535fc6345a0`.

Adapter function invocation hook payloads.

### `AdapterFunctionInvocationCompletePayload`

*class* — [line 13](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/hooks/adapter_function.py#L13) (`MelleaBasePayload`)

Payload for `adapter_function_invocation_complete` — after an adapter function invocation finishes.

### `AdapterFunctionPhaseCompletePayload`

*class* — [line 40](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/hooks/adapter_function.py#L40) (`MelleaBasePayload`)

Payload for `adapter_function_phase_complete` — after one lifecycle phase finishes.

---

## Module `mellea.plugins.hooks.component`

Source: [`mellea/plugins/hooks/component.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/hooks/component.py) at commit `a535fc6345a0`.

Component lifecycle hook payloads.

### `ComponentPreExecutePayload`

*class* — [line 13](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/hooks/component.py#L13) (`MelleaBasePayload`)

Payload for `component_pre_execute` — before component execution via `aact()`.

### `ComponentPostSuccessPayload`

*class* — [line 39](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/hooks/component.py#L39) (`MelleaBasePayload`)

Payload for `component_post_success` — after successful component execution.

### `ComponentPostErrorPayload`

*class* — [line 70](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/hooks/component.py#L70) (`MelleaBasePayload`)

Payload for `component_post_error` — after component execution fails.

---

## Module `mellea.plugins.hooks.generation`

Source: [`mellea/plugins/hooks/generation.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/hooks/generation.py) at commit `a535fc6345a0`.

Generation pipeline hook payloads.

### `GenerationPreCallPayload`

*class* — [line 14](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/hooks/generation.py#L14) (`MelleaBasePayload`)

Payload for `generation_pre_call` — before LLM backend call.

### `GenerationPostCallPayload`

*class* — [line 38](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/hooks/generation.py#L38) (`MelleaBasePayload`)

Payload for `generation_post_call` — fires once the model output is fully computed.

### `GenerationErrorPayload`

*class* — [line 64](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/hooks/generation.py#L64) (`MelleaBasePayload`)

Payload for `generation_error` — fires when the LLM backend raises an exception.

### `GenerationBatchPreCallPayload`

*class* — [line 86](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/hooks/generation.py#L86) (`MelleaBasePayload`)

Payload for `generation_batch_pre_call` — fires once before a batch generation request.

### `GenerationBatchPostCallPayload`

*class* — [line 116](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/hooks/generation.py#L116) (`MelleaBasePayload`)

Payload for `generation_batch_post_call` — fires once after a batch generation succeeds.

### `GenerationBatchErrorPayload`

*class* — [line 141](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/hooks/generation.py#L141) (`MelleaBasePayload`)

Payload for `generation_batch_error` — fires once when a batch generation request fails.

---

## Module `mellea.plugins.hooks.sampling`

Source: [`mellea/plugins/hooks/sampling.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/hooks/sampling.py) at commit `a535fc6345a0`.

Sampling pipeline hook payloads.

### `SamplingLoopStartPayload`

*class* — [line 13](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/hooks/sampling.py#L13) (`MelleaBasePayload`)

Payload for `sampling_loop_start` — when sampling strategy begins.

### `SamplingIterationPayload`

*class* — [line 36](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/hooks/sampling.py#L36) (`MelleaBasePayload`)

Payload for `sampling_iteration` — after each sampling attempt.

### `SamplingRepairPayload`

*class* — [line 65](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/hooks/sampling.py#L65) (`MelleaBasePayload`)

Payload for `sampling_repair` — when repair is invoked after validation failure.

### `SamplingLoopEndPayload`

*class* — [line 90](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/hooks/sampling.py#L90) (`MelleaBasePayload`)

Payload for `sampling_loop_end` — when sampling completes.

---

## Module `mellea.plugins.hooks.session`

Source: [`mellea/plugins/hooks/session.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/hooks/session.py) at commit `a535fc6345a0`.

Session lifecycle hook payloads.

### `SessionPreInitPayload`

*class* — [line 16](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/hooks/session.py#L16) (`MelleaBasePayload`)

Payload for `session_pre_init` — before backend initialization.

### `SessionPostInitPayload`

*class* — [line 33](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/hooks/session.py#L33) (`MelleaBasePayload`)

Payload for `session_post_init` — after session is fully initialized.

### `SessionResetPayload`

*class* — [line 47](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/hooks/session.py#L47) (`MelleaBasePayload`)

Payload for `session_reset` — when session context is reset.

### `SessionCleanupPayload`

*class* — [line 58](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/hooks/session.py#L58) (`MelleaBasePayload`)

Payload for `session_cleanup` — before session cleanup/teardown.

---

## Module `mellea.plugins.hooks.streaming`

Source: [`mellea/plugins/hooks/streaming.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/hooks/streaming.py) at commit `a535fc6345a0`.

Streaming pipeline hook payloads.

### `StreamingStartPayload`

*class* — [line 13](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/hooks/streaming.py#L13) (`MelleaBasePayload`)

Payload for `streaming_start` — before a `stream_with_chunking` run starts.

### `StreamingEventPayload`

*class* — [line 31](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/hooks/streaming.py#L31) (`MelleaBasePayload`)

Payload for `streaming_event` — fired once per `StreamEvent`.

### `StreamingEndPayload`

*class* — [line 47](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/hooks/streaming.py#L47) (`MelleaBasePayload`)

Payload for `streaming_end` — when `stream_with_chunking` finishes.

### `StreamingOrchestrationStartPayload`

*class* — [line 75](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/hooks/streaming.py#L75) (`MelleaBasePayload`)

Payload for `streaming_orchestration_start` — on the orchestration task, before the stream is drained.

### `StreamingOrchestrationEndPayload`

*class* — [line 85](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/hooks/streaming.py#L85) (`MelleaBasePayload`)

Payload for `streaming_orchestration_end` — on the orchestration task, after the stream is drained.

---

## Module `mellea.plugins.hooks.tool`

Source: [`mellea/plugins/hooks/tool.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/hooks/tool.py) at commit `a535fc6345a0`.

Tool execution hook payloads.

### `ToolPreInvokePayload`

*class* — [line 13](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/hooks/tool.py#L13) (`MelleaBasePayload`)

Payload for `tool_pre_invoke` — before tool/function invocation.

### `ToolPostInvokePayload`

*class* — [line 31](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/hooks/tool.py#L31) (`MelleaBasePayload`)

Payload for `tool_post_invoke` — after tool execution.

---

## Module `mellea.plugins.hooks.validation`

Source: [`mellea/plugins/hooks/validation.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/hooks/validation.py) at commit `a535fc6345a0`.

Validation hook payloads.

### `ValidationPreCheckPayload`

*class* — [line 13](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/hooks/validation.py#L13) (`MelleaBasePayload`)

Payload for `validation_pre_check` — before requirement validation.

### `ValidationPostCheckPayload`

*class* — [line 34](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/plugins/hooks/validation.py#L34) (`MelleaBasePayload`)

Payload for `validation_post_check` — after validation completes.

---

*Generated from source by `docs/scripts/generate_package_map.py` — do not edit by hand; regenerate instead. Symbols and annotations reflect the pinned commit exactly; where type annotations are absent in source, this page says so rather than guessing.*
