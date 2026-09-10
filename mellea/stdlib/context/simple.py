# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stateless single-turn context (no history is forwarded to the model)."""

from __future__ import annotations

from mellea.core import Context, Span


class SimpleContext(Context):
    """A `SimpleContext` is a context in which each interaction is a separate and independent turn. The history of all previous turns is NOT saved..

    Note:
        Because `view_for_generation` always returns an empty list, anything
        passed to `SimpleContext.add` is **never forwarded to the model** — it
        is recorded only on the in-memory context chain. The action passed to
        `generate_from_context` (or `MelleaSession.chat`) is the *only* thing
        that reaches the model. Combining `.add(...)` with an empty/whitespace
        action therefore produces an empty user prompt; the OpenAI, LiteLLM,
        and Ollama backends now reject such calls with a `ValueError` (see
        issue #1597) rather than sending an empty conversation to the model.
    """

    def add(self, c: Span) -> SimpleContext:
        """Add a new component or CBlock to the context and return the updated context.

        The added span is stored on the context chain but is **not forwarded
        to the model on subsequent generations** — `SimpleContext.view_for_generation`
        always returns an empty list, so each generation is treated as a
        stateless, independent turn. To actually talk to the model, pass the
        prompt as the `action` argument to `MelleaSession.chat` /
        `Backend.generate_from_context`, not via `add`.

        Args:
            c (Span): The component, content
                block, or model output to record.

        Returns:
            SimpleContext: A new `SimpleContext` containing only the added entry;
            prior history is not retained.
        """
        return SimpleContext.from_previous(self, c)

    def view_for_generation(self) -> list[Span] | None:
        """Return an empty list, since `SimpleContext` does not pass history to the model.

        Each call to the model is treated as a stateless, independent exchange.
        No prior turns are forwarded. Spans recorded via `add` are kept on
        the in-memory chain (`as_list`) for inspection but discarded for
        generation.

        Returns:
            list[Span] | None: Always an empty list.
        """
        return []
