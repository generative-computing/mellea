---
title: "Adapter functions"
description: "Adapter-accelerated RAG quality checks using LoRA/aLoRA adapters with Granite models."
# diataxis: how-to
---

**Prerequisites:** use `uv sync --extra hf` for runtime LoRA/aLoRA adapter
functions and local [Granite Switch](../reference/glossary.md#granite-switch)
checkpoints. Both local paths require a GPU or Apple Silicon Mac. An
OpenAIBackend using a Granite Switch model served via vLLM uses
`uv sync --extra switch` when it downloads embedded adapter metadata.

Adapter functions are adapter-accelerated operations for RAG quality checks. Their
LoRA/aLoRA weights are loaded locally, selected from an embedded checkpoint, or
bundled into an Ollama model — faster and more reliable than prompting a
general-purpose model for these specialised micro-tasks.

> **Backend note:** Adapter functions work with three backends:
>
> - **LocalHFBackend** — loads LoRA/aLoRA adapters from the catalog at runtime.
>   A local Granite Switch checkpoint can instead use
>   `load_embedded_adapters=True`; install `mellea[hf]` first. Only
>   adapter functions embedded in the checkpoint are then available. Requires a
>   GPU or Apple Silicon Mac.
> - **OpenAIBackend** — uses a Granite Switch model served via vLLM with
>   `load_embedded_adapters=True`. Only adapter functions embedded in the model are
>   available — check the model's `adapter_index.json` for the list.
>   See `docs/docs/examples/granite-switch/README.md`
> - **OllamaModelBackend** — uses an Ollama model that bundles the adapter.
>   Ollama bundles one adapter per model, so pass
>   `adapter_models={"uncertainty": "<tag>", ...}` to route each adapter function
>   to its model. Install `mellea[switch]` to download the adapter's `io.yaml`.
>
> Adapter functions do not work with other remote backends.

Set up the backend once and reuse it across adapter function calls:

```python
# Requires: mellea[hf]
# Returns: LocalHFBackend
from mellea.backends.huggingface import LocalHFBackend

backend = LocalHFBackend(model_id="ibm-granite/granite-4.1-3b")
```

## Use an adapter bundled in an Ollama model

Ollama serves adapter weights as part of a model tag; Mellea does not load the
weights separately. For local development, build a bundled uncertainty model
from the pinned official Granite base and adapter artefacts:

```bash
MELLEA_OLLAMA_UNCERTAINTY_MODEL="$(
  ./test/scripts/build_ollama_uncertainty_adapter.sh
)"
export MELLEA_OLLAMA_UNCERTAINTY_MODEL
```

Install the lightweight Hugging Face Hub dependency that retrieves the
catalogued `io.yaml`:

```bash
uv sync --extra switch
```

For one adapter function, use the bundled tag for both ordinary chat and the
adapter route. Before the invocation tokens appear, the aLoRA model behaves as
the base model; keeping one model identity lets Ollama reuse its own prefix
cache where available.

```python
import os

from mellea.backends import ModelOption
from mellea.backends.ollama import OllamaModelBackend
from mellea.stdlib.components import Message
from mellea.stdlib.components.intrinsic import core
from mellea.stdlib.context import ChatContext

backend = OllamaModelBackend(
    model_id=os.environ["MELLEA_OLLAMA_UNCERTAINTY_MODEL"],
    adapter_base_model_name="granite-4.1-3b",
    model_options={ModelOption.CONTEXT_WINDOW: 4096},
    adapter_models={
        "uncertainty": os.environ["MELLEA_OLLAMA_UNCERTAINTY_MODEL"],
    },
)
context = (
    ChatContext()
    .add(Message("user", "What is the square root of 4?"))
    .add(Message("assistant", "The square root of 4 is 2."))
)

print(core.check_certainty(context, backend))
```

When an application needs several adapter functions, keep the base model as
`model_id` and map each function to its bundled model tag. Current Ollama
model packaging exposes one adapter per tag, so calls across those tags do not
share a KV cache. Use a Granite Switch checkpoint when multi-adapter,
single-model serving is important.

See `docs/examples/intrinsics/uncertainty_ollama.py` for the complete
executable example.

## Use a local Granite Switch checkpoint

Granite Switch checkpoints contain their adapter functions already. Pass the
checkpoint to `LocalHFBackend` with `load_embedded_adapters=True`; existing
helper functions such as `rag.check_answerability()` work unchanged.

```python
# Requires: mellea[hf]
# Returns: LocalHFBackend
from mellea.backends.huggingface import LocalHFBackend
from mellea.backends.model_ids import IBM_GRANITE_SWITCH_4_1_3B_PREVIEW

backend = LocalHFBackend(
    model_id=IBM_GRANITE_SWITCH_4_1_3B_PREVIEW,
    load_embedded_adapters=True,
)
```

Only adapter functions listed in the checkpoint's `adapter_index.json` are
available. Mellea warns once if Granite Switch's installed package metadata
does not yet include Mellea's resolved Transformers version; the warning
disappears after Granite Switch publishes compatible metadata.

Or, with a Granite Switch model via the OpenAI backend:

```python
from mellea.backends.openai import OpenAIBackend
from mellea.backends.model_ids import IBM_GRANITE_SWITCH_4_1_3B_PREVIEW
from mellea.formatters import TemplateFormatter

backend = OpenAIBackend(
    model_id=IBM_GRANITE_SWITCH_4_1_3B_PREVIEW.hf_model_name,
    formatter=TemplateFormatter(model_id=IBM_GRANITE_SWITCH_4_1_3B_PREVIEW.hf_model_name),
    base_url="http://localhost:8000/v1",  # vLLM server
    api_key="EMPTY",
    load_embedded_adapters=True,
)
```

## Decisions: typed one-token answers

Some Granite Switch checkpoints embed a **decision** adapter function (named
`decision`). It answers a fixed-answer question about some state
with a probability, a choice, or a score — read from the model's first
generated token, so a decision costs one forward pass and is fast enough for
control flow: `if` on a probability, route on a choice, threshold a score.

### Serving

Serve a checkpoint such as `ibm-granite/granite-switch-4.1-8b` via vLLM
with `--max-logprobs 20` (the readout requests up to 20), installing the
[granite-switch](https://pypi.org/project/granite-switch/) package alongside
vLLM (the `GraniteSwitchForCausalLM` architecture is plugin-registered, not
in upstream vLLM). On vLLM 0.30.0 and later the backend also asks the server
for the logprobs of the exact label token ids (`logprob_token_ids`, ids taken
from the server's `/tokenize`), so a label outside the top 20 cannot be
missed. Other servers, and older vLLM, keep the top-20 request. Set up the
backend as in
[Use a local Granite Switch checkpoint](#use-a-local-granite-switch-checkpoint).

Decisions also work against [llama.cpp](https://github.com/ggml-org/llama.cpp)'s
`llama-server` and the HuggingFace backend (`LocalHFBackend` with
`granite-switch[hf]>=0.1.1`). Runnable examples:
[`decisions.py`](../../examples/intrinsics/decisions.py) (vLLM) and
[`decisions_llamacpp.py`](../../examples/intrinsics/decisions_llamacpp.py).

Both serving backends accept `adapter_revision` to pin the Hub revision
used for embedded-adapter discovery. Pin it to the same commit your
server loaded, so a mid-flight Hub update cannot silently change the
adapter configuration your client reads:

```python
backend = OpenAIBackend(
    model_id="ibm-granite/granite-switch-4.1-8b",
    base_url="http://localhost:8000/v1",
    load_embedded_adapters=True,
    adapter_revision="<commit-sha>",
)
```

For a step-by-step tutorial covering a realistic triage workflow,
custom true/false descriptions, conversational state, and error
handling, see [Tutorial: One-Token
Decisions](../tutorials/09-one-token-decisions.md).

### The API

```python
from mellea.stdlib.components.intrinsic import decisions

# Predicate: a probability you threshold. Ordering operators delegate to
# .probability, so this reads naturally:
if decisions.decide(ticket, backend, "Does this require urgent attention?") >= 0.7:
    escalate(ticket)

# Choice: the best option key plus the distribution.
route = decisions.choose(
    ticket,
    backend,
    "Which team handles this?",
    {"payments": "Payout failures and refunds", "account": "Login and account access"},
)
route.choice        # "payments"
route.probabilities # {"payments": 0.91, "account": 0.09}

# Score: expected level (may land between levels) plus per-level probabilities.
sev = decisions.score(ticket, backend, "How urgent is this?",
                      ["can wait a week", "this week", "today"])
sev.score           # 1.7
sev.probabilities   # {"0": 0.05, "1": 0.20, "2": 0.75}
```

`state` (the first argument) is the thing being judged: a string, a dict or
list (rendered as JSON), or a `ChatContext` for conversational state. A
`ChatContext` forwards its own messages as the state (no separate state turn
is rendered for it); any other `Context` type raises a `TypeError`. A
conversational state sends the conversation's turns as they are, without the
`State:` header that plain data gets. The adapter was trained on a single
block of state, so check long or multi-turn conversations on your own
examples. A Pydantic model or dataclass is rendered as the JSON of its
fields.
All three helpers take `model_options`, but every decision-critical
setting (`max_completion_tokens=1`, greedy `temperature=0.0`, `top_p` and
`top_k`, logprobs, the sized `top_logprobs`, no streaming) is pinned by
the capability contract. Logit-shaping options (`logit_bias`, the
frequency, presence and repetition penalties, a custom `logits_processor`)
are rejected too. A conflicting option raises rather than silently
breaking the readout, whether it is passed per call or set on the backend.
`enable_thinking` is the exception: the decision template has no thinking
branch, so any value you set is dropped and the call proceeds.
Special-token strings in your text are split with an invisible word
joiner, so state, question and option text cannot forge a turn or switch
the adapter. That covers the `<|name|>` family (role markers, end of text,
adapter control tokens) and the tag tokens of the Switch checkpoints
(`<tool_call>`, `<think>`, `<schema>` and the like), so tool and reasoning
markup in a state is read as plain text. Ordinary text is unchanged.
On `LocalHFBackend` the label probabilities are read from the raw
logits at the label token ids, so no generation option can move them. The
probabilities are the model's own and are not calibrated against ground
truth; fit a calibration on labelled data before acting on a threshold.
`choose` takes 2-10 options and `score` 2-10
levels. When serving with vLLM, do not enable
`--logprobs-mode processed_logprobs`; the decision readout expects
raw logprobs, and applying temperature before returning them would
distort the probabilities.

### How it works

The adapter is trained on a specific prompt layout: a fixed system prompt
("You are a decision engine…"), a state turn (`State:\n<your state>`), and
a decision turn (`### DECISION\nQuestion:\n…\nOptions:\nA: …\nB: …\n…`).
Mellea assembles this layout from the capability's configuration and sends
it to the model, which generates exactly one token — the answer label
(`A`, `B`, `C`, …).

The answer is decoded from that token's logprob distribution: the top-20
logprobs are read from the response, the option labels are matched by exact
token identity, and a restricted softmax over the used labels produces the
probability distribution. For `decide`, the labels map to
`[false, true]` (so `.probability` is P(true)); for `choose`, the
argmax label maps back to the option key; for `score`, the expected level
is computed from the per-level probabilities.

The capability's prompt layout is carried by Mellea as an interim
configuration, merged field by field into the checkpoint's published
`io.yaml` at adapter discovery: a field the published file omits gets the
carried value, and a field it publishes wins over the carried one (with a
warning naming both). The merged configuration is validated at discovery;
if it is invalid (for example, a published field breaks a placeholder the
rewriter depends on), the decision capability is not registered and the
error from `decide`/`choose`/`score` says why.

### Answer semantics

A bare `if answer:` is always `True` — test `answer.probability`, compare
`answer.choice == "..."`, or compare a `PredicateAnswer`/`ScoreAnswer`
directly against a number (ordering operators delegate to the primary
field). `confidence` on choice and score answers is an uncalibrated
distribution-shape statistic, not a probability of correctness. For a
choice it is the top probability rescaled so a uniform distribution reads 0
and certainty reads 1, `(max(p) − 1/K) / (1 − 1/K)`. For a score it is 1
minus the expected distance from the most likely level, divided by the mean
distance every level has from the scale midpoint. Both follow the OpenAI
decisions specification's formulas, and both reproduce its worked examples.

### Failure modes

Failure is loud, never silent:

- **Label-mass floor**: if the answer distribution does not concentrate on
  the declared options (mass < 0.5), the call raises. This usually indicates a serving or template problem.
- **Missing labels**: if the server cannot return enough logprobs to see
  all option labels, the call raises with a server-configuration hint
  (vLLM: check `--max-logprobs`).
- **Marker contamination**: if the invocation marker (`### DECISION`)
appears in the state, question, or options text (not just the decision
turn), the call raises `AloraActivationError` — the adapter would activate
  at the wrong position.
- **Parameter conflicts**: a caller `model_options` key that conflicts
  with a pinned decision-critical setting (e.g. `MAX_NEW_TOKENS: 64`)
  raises rather than silently breaking the readout.

Every decision is logged with its question, answer distribution, label
mass, and latency; set `MELLEA_LOGS_LEVEL=DEBUG` to also log option
descriptions and the full state.

## Answerability

Check whether a set of retrieved documents can answer a given question:

```python
# Requires: mellea[hf]
# Returns: bool
from mellea.backends.huggingface import LocalHFBackend
from mellea.stdlib.components import Document, Message
from mellea.stdlib.components.intrinsic import rag
from mellea.stdlib.context import ChatContext

backend = LocalHFBackend(model_id="ibm-granite/granite-4.1-3b")
context = ChatContext().add(Message("assistant", "Hello! How can I help you?"))
question = "What is the square root of 4?"

docs_answerable = [Document("The square root of 4 is 2.")]
docs_not_answerable = [Document("The square root of 8 is approximately 2.83.")]

print(rag.check_answerability(question, docs_answerable, context, backend))   # True
print(rag.check_answerability(question, docs_not_answerable, context, backend))  # False
```

## Hallucination detection

Flag sentences in an assistant response that are not grounded in the source documents:

```python
# Requires: mellea[hf]
# Returns: list[str]
from mellea.backends.huggingface import LocalHFBackend
from mellea.stdlib.components import Document, Message
from mellea.stdlib.components.intrinsic import rag
from mellea.stdlib.context import ChatContext

backend = LocalHFBackend(model_id="ibm-granite/granite-4.1-3b")
context = (
    ChatContext()
    .add(Message("assistant", "Hello! How can I help you?"))
    .add(Message("user", "Tell me about yellow fish."))
)

response = "Purple bumble fish are yellow. Green bumble fish are also yellow."
documents = [
    Document(doc_id="1", text="The only type of fish that is yellow is the purple bumble fish.")
]

result = rag.flag_hallucinated_content(response, documents, context, backend)
print(result)
# Flags "Green bumble fish are also yellow." as hallucinated
```

## Answer relevance rewriting

Rewrite a vague or incomplete answer to be more grounded in the source documents:

```python
# Requires: mellea[hf]
# Returns: str
from mellea.backends.huggingface import LocalHFBackend
from mellea.stdlib.components import Document, Message
from mellea.stdlib.components.intrinsic import rag
from mellea.stdlib.context import ChatContext

backend = LocalHFBackend(model_id="ibm-granite/granite-4.1-3b")
context = ChatContext().add(Message("user", "Who attended the meeting?"))
documents = [
    Document("Meeting attendees: Alice, Bob, Carol."),
    Document("Meeting time: 9:00 am to 11:00 am."),
]
original = "Many people attended the meeting."

result = rag.rewrite_answer_for_relevance(original, documents, context, backend)
print(result)
# A more specific, grounded answer — output will vary
```

## Query rewriting

Rewrite an ambiguous user query using conversation history to improve retrieval:

```python
# Requires: mellea[hf]
# Returns: str
from mellea.backends.huggingface import LocalHFBackend
from mellea.stdlib.components import Message
from mellea.stdlib.components.intrinsic import rag
from mellea.stdlib.context import ChatContext

backend = LocalHFBackend(model_id="ibm-granite/granite-4.1-3b")
context = (
    ChatContext()
    .add(Message("assistant", "Welcome to pet questions!"))
    .add(Message("user", "I have two pets: a dog named Rex and a cat named Lucy."))
    .add(Message("assistant", "Rex spends a lot of time outdoors, and Lucy is always inside."))
    .add(Message("user", "Sounds good! Rex must love exploring outside."))
)
next_turn = "But is he more likely to get fleas because of that?"

result = rag.rewrite_question(next_turn, context, backend)
print(result)
# Resolves "he" to "Rex" and incorporates context about outdoor exposure
```

## Citations

Find supporting sentences in source documents for a given assistant response:

```python
# Requires: mellea[hf]
# Returns: dict
from mellea.backends.huggingface import LocalHFBackend
from mellea.stdlib.components import Document, Message
from mellea.stdlib.components.intrinsic import rag
from mellea.stdlib.context import ChatContext

backend = LocalHFBackend(model_id="ibm-granite/granite-4.1-3b")
context = ChatContext().add(
    Message("user", "How did Murdoch expand in Australia versus New Zealand?")
)
response = (
    "Murdoch expanded in Australia and New Zealand by acquiring local newspapers. "
    "I do not have information about his expansion in New Zealand after purchasing "
    "The Dominion."
)
documents = [
    Document(doc_id="1", text="Keith Rupert Murdoch was born on 11 March 1931 in Melbourne..."),
    Document(doc_id="2", text="This document has nothing to do with Rupert Murdoch."),
]

result = rag.find_citations(response, documents, context, backend)
print(result)
# Maps each response sentence to supporting document sentences
```

## Direct adapter function usage

> **Advanced:** For custom adapter tasks, compose an `Adapter` directly from
> an `Identity`, an output contract, and a weights binding.

```python
# Requires: mellea[hf]
# Returns: dict
import mellea.stdlib.functional as mfuncs
from mellea.backends.adapters import Adapter, Identity, LocalFileBinding, get_io_contract
from mellea.backends.adapters.catalog import AdapterType, fetch_intrinsic_metadata
from mellea.backends.huggingface import LocalHFBackend
from mellea.stdlib.components import Intrinsic, Message
from mellea.stdlib.context import ChatContext

backend = LocalHFBackend(model_id="ibm-granite/granite-4.1-3b")

# Compose an adapter by task name — get_io_contract returns the catalog's
# declared contract (or a permissive fallback for a name outside it).
# requirement-check's catalog entry lists LoRA before aLoRA, so
# LocalFileBinding.from_catalog would pick LoRA — build the binding
# explicitly instead when you want the aLoRA variant specifically; identity
# and weights must agree, since nothing currently cross-checks them.
_metadata = fetch_intrinsic_metadata("requirement-check")
req_adapter = Adapter(
    identity=Identity(
        name="requirement-check",
        adapter_type="alora",
        capability=_metadata.effective_capability,
    ),
    io_contract=get_io_contract("requirement-check"),
    weights=LocalFileBinding(
        name="requirement-check",
        adapter_type=AdapterType.ALORA,
        repo_id=_metadata.repo_id,
        revision=_metadata.revision,
    ),
)
backend.add_adapter(req_adapter)

ctx = ChatContext()
ctx = ctx.add(Message("user", "Hi, can you help me?"))
ctx = ctx.add(Message("assistant", "Yes! What can I help with?"))

out, _ = mfuncs.act(
    Intrinsic(
        "requirement-check",
        intrinsic_kwargs={"requirement": "The assistant is helpful."},
    ),
    ctx,
    backend,
)
print(out)  # {"requirement_check": {"score": 1.0}}
```

The `Intrinsic` component loads aLoRA adapters (falling back to LoRA) by task name.
For OpenAI backends with Granite Switch, adapters are loaded from the model's
Hugging Face repository configuration instead of the adapter function catalog.
Output format is task-specific — `requirement-check` returns `{"requirement_check": {"score": <float>}}`.

For a fully custom, non-catalog adapter — your own trained LoRA/aLoRA weights,
not one of the built-in adapter functions — see
[Adding a custom adapter function in 20 lines](../tutorials/07-custom-adapter-function.md).

## Composable adapter construction (advanced)

`Adapter` composes an `Identity`, an output contract (`IOContract`), and a
weights binding into a single, inspectable object. Both `LocalHFBackend` and
`OpenAIBackend` accept a composed `Adapter` directly via `add_adapter` —
dispatching on the weights binding's reality (`LocalFileBinding` for
LocalFile/PEFT, `EmbeddedBinding` for Embedded/Granite Switch) — alongside
the deprecated shim classes, which remain functional for now (Epic #929,
issue #1144). An `EmbeddedBinding` adapter additionally requires `add_adapter`'s
`config=` argument (the raw io.yaml mapping) — `add_adapter` raises `ValueError`
without it, since that reality's config cannot be cheaply re-derived later; see
below.

Each weights binding models how its deployment turns an adapter on.
`LocalFileBinding` downloads and loads LoRA/aLoRA weights, so it exposes a
`prepare`/`activate`/`deactivate`/`release` lifecycle — driven automatically
by `add_adapter`, so a caller need not call `prepare()` itself:

```python
# Requires: mellea[hf]
from mellea.backends.adapters import Adapter, Identity, LocalFileBinding, get_io_contract
from mellea.backends.huggingface import LocalHFBackend

# LocalFile/PEFT reality — LocalHFBackend downloads and loads the weights.
hf_backend = LocalHFBackend(model_id="ibm-granite/granite-4.1-3b")
hf_adapter = Adapter(
    identity=Identity(name="answerability", adapter_type="lora"),
    io_contract=get_io_contract("answerability"),
    weights=LocalFileBinding.from_catalog("answerability"),
)
hf_backend.add_adapter(hf_adapter)  # downloads and loads the weights
```

`EmbeddedBinding` has no weights to manage — the adapter is already part of
the served base model — so it exposes a single method, `apply_activation`,
that edits the outgoing request instead of a lifecycle. Its `io.yaml` config
comes from the served checkpoint's `adapter_index.json`, not from anything
you can construct by hand, so registration goes through
`register_embedded_adapter_model` (or `resolve_adapter`) rather than a bare
`add_adapter(adapter)` call with no `config=`:

```python
from mellea.backends.openai import OpenAIBackend
from mellea.backends.model_ids import IBM_GRANITE_SWITCH_4_1_3B_PREVIEW

switch_backend = OpenAIBackend(
    model_id=IBM_GRANITE_SWITCH_4_1_3B_PREVIEW.hf_model_name,
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
    load_embedded_adapters=False,
)
# Discovers "answerability" from the model's Hugging Face repo and composes
# an Adapter (Identity + IOContract + EmbeddedBinding) for it, including the
# io.yaml config a bare Adapter(weights=EmbeddedBinding.from_base_model(...))
# construction has no way to supply.
switch_backend.register_embedded_adapter_model(
    IBM_GRANITE_SWITCH_4_1_3B_PREVIEW.hf_model_name, intrinsic_name="answerability"
)
```

Weights-binding support by backend today:

| Backend | `LocalFileBinding` (LocalFile/PEFT) | `EmbeddedBinding` (Embedded/Granite Switch) | `ServerMediatedBinding` |
| --- | --- | --- | --- |
| `LocalHFBackend` | ✅ shipping — `add_adapter` accepts a composed `Adapter` or a bare `LocalFileBinding` directly | ✅ shipping — `load_embedded_adapters=True`, or `add_adapter(adapter, config=...)`/`register_embedded_adapter_model` with a composed `Adapter` | — |
| `OpenAIBackend` | — | ✅ shipping — `load_embedded_adapters=True`, or `add_adapter(adapter, config=...)`/`register_embedded_adapter_model` with a composed `Adapter` | — |
| `OllamaModelBackend` | — | — | ✅ model selection through `adapter_models` for catalogued adapter functions; lifecycle telemetry is tracked separately |

`ServerMediatedBinding` currently supports Ollama's bundled-model path. A full
server-mediated lifecycle and telemetry contract is tracked separately.
Discovering *multiple* embedded adapters from a Granite Switch checkpoint or
Hub repo (rather than one already-known name) still goes through
`register_embedded_adapter_model`, which builds the composed `Adapter`
instances for you.

---

## Guardian adapter functions

Safety and factuality checks use a separate set of Guardian-specific adapter functions:
`guardian_check()`, `policy_guardrails()`, `factuality_detection()`, and
`factuality_correction()`. These are documented in the
[Safety Guardrails](../how-to/safety-guardrails.md) how-to guide.

**See also:**
[Adding a custom adapter function in 20 lines](../tutorials/07-custom-adapter-function.md) |
[Handling a breaking adapter schema change](../tutorials/08-adapter-schema-migrations.md) |
[Adapter function metrics](../observability/metrics.md#adapter-function-metrics)
