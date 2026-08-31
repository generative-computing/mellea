---
title: "OrcaRouter"
description: "Use OrcaRouter — an OpenAI-compatible AI gateway with adaptive routing, failover, and gateway-level security — through Mellea's named OrcaRouterBackend."
sidebar_label: "OrcaRouter"
# diataxis: how-to
---

[`OrcaRouterBackend`](../reference/glossary.md#backend) connects Mellea to the
[OrcaRouter](https://www.orcarouter.ai) gateway — a single OpenAI-compatible
endpoint in front of many hosted models. Like OpenAI, OrcaRouter exposes a
provider/model namespace across many models; on top of that it adds adaptive
routing, automatic failover, zero-markup inference, observability, guardrails,
and agent-tool governance behind the same endpoint. Because the endpoint is
OpenAI-compatible, Mellea reuses its full OpenAI backend machinery — structured
output, tool calling, thinking-mode handling, and streaming all work unchanged.

**Prerequisites:** `pip install mellea`, an [OrcaRouter](https://www.orcarouter.ai) API key.

## Setup

Set your API key as an environment variable (recommended):

```bash
export ORCAROUTER_API_KEY=sk-orca-...
```

## Using the named backend

`start_session()` accepts `backend_name="orcarouter"` and defaults to OrcaRouter's
adaptive-routing model, which grades each prompt and routes to the best frontier
or open-weights model:

```python
# Requires: mellea
# Returns: ModelOutputThunk
import mellea

m = mellea.start_session(
    backend_name="orcarouter",
    model_id="orcarouter/auto",
    context_type="chat",
)
reply = m.chat("What is the capital of France?")
print(str(reply))
# Output will vary — LLM responses depend on model and temperature.
```

Or construct the backend directly:

```python
# Requires: mellea
# Returns: MelleaSession
from mellea import MelleaSession
from mellea.backends.orcarouter import OrcaRouterBackend

m = MelleaSession(
    OrcaRouterBackend(model_id="orcarouter/auto"),
    ctx=ChatContext(),
)
```

`OrcaRouterBackend` reads `ORCAROUTER_API_KEY` from the environment and defaults
`base_url` to `https://api.orcarouter.ai/v1`. Pass the key or a different base
URL directly to override either:

```python
# Requires: mellea
# Returns: MelleaSession
from mellea import MelleaSession
from mellea.backends.orcarouter import OrcaRouterBackend

m = MelleaSession(
    OrcaRouterBackend(
        model_id="orcarouter/auto",
        api_key="sk-orca-...",
    ),
)
```

## Available models

OrcaRouter's `orcarouter/auto` routes each prompt to the best-suited model. You
can also target a specific model by name; pass any model string OrcaRouter
serves to the `model_id` argument. The `ModelIdentifier` constant
`model_ids.ORCAROUTER_AUTO` carries the auto-routing name for type-checked use.

## Security and governance

OrcaRouter runs gateway-level, zero-trust security for AI agents on the same
endpoint — screening every prompt and response and governing every tool call on
a default-deny basis, with no application code changes. Mellea's tool calling and
structured-output features are forwarded as normal OpenAI-compatible requests, so
the same Mellea code gains those safeguards when pointed at OrcaRouter.

## Troubleshooting

### `ORCAROUTER_API_KEY` not set error

Either export the environment variable or pass `api_key` directly to
`OrcaRouterBackend`.

### Model not found

The model string must exactly match a model name OrcaRouter serves. Use
`orcarouter/auto` for adaptive routing, or list models from OrcaRouter's API.

---

**See also:** [Backends and Configuration](../how-to/backends-and-configuration) | [OpenAI and OpenAI-Compatible APIs](./openai)
