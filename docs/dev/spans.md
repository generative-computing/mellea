# Design Document for Spans

> **Status: unfinished stub.** This note breaks off mid-decision ("We need to
> document which approach we choose...") and no decision was ever recorded; no
> `SpanContext` or `__span_repr__` exists in the codebase. The `/dev/spans` URL
> on the published docs site now redirects to the unrelated OpenTelemetry
> tracing page ([`docs/docs/observability/tracing.md`](../docs/observability/tracing.md)),
> which does not cover this design. See #1482 for the `docs/dev/` currency
> discussion.

## Span Contexts

We will introduce a SpanContext which will behave kind of like a heap but with transformer-running-on-GPU memory primitives instead of malloc/realloc/free. The public interface to a SpanContext will roughly correspond to the sort of stuff you can do in Span algebras, if you've seen some of that work.

## Mapping STDLIB to Spans

There are two broad philosophies to choose from for Spans.

### The Span Representation Approach

All Components and CBlocks get a __span_repr__ which maps the all things to a Span representation. The Component owner is responsible for saying how something gets represented as a Span, and is also responsible for defining caching boundaries (via a cache_boundary tag).

### The Span Formatter Approach

There is a Formatter which maps Components and CBlocks to Spans, as a pure function. Similar to how the TemplateFormatter works today.

We need to document which approach we choose and discuss why it was chosen.

