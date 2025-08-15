# Constrained Decoding

## How do constraints get defined?

Should we be thinking bigger than pydantic? Should it be possible to pass arbitrary grammars? If so, what's the abstract interface for those? Should this be factored out into llm-io?

## How do constraints get passed around?

The `m` framework currently uses the `format` argument to pydantic schemas, **outside of model args**. Should we be using `@@@format@@@` within ModelArgs instead? Hendrik describes the behavior of model args like this (paraghased by Nathan):

> If a keyword had meaning across multiple types of backends, and if it means the same thing in all of those backends but has different names, then we use the `@@@`-style args so that the user can pass these args across all backends in the same way. Otherwise, the arguments in model_args are passed along verbatim.

This argues for `@@@format@@@` as opposed to a dedicated `format` option in the method signature. Or, in the alternative, for an entire re-think of ModelArgs.

## Integration with grammar-targeted LLMs

Some LLMs target generation in a particular grammar. Examples include:
 * ALoRAs that target very simple grammars
 * code generatorrs that target particular PLs
 * models (or model modes) tuned to generate JSON
 * models (or model modes) tuned to generate YAML or particular fragments of YAML (such as k8s configs)

Should we be doing constrained decoding in these cases, or should we treat deviation from the grammar as an exception? Probably the answer is "it depends". Masataro had a nice idea of **taking the sum of logits of grammatically feasible completions** and ensuring that this sum is above some threshold. How would supporting this change the interface described in the "How do constraints get defined?" section?