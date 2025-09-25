"""This module holds shim backends used for smoke tests."""

from mellea.backends import Backend, BaseModelSubclass
from mellea.stdlib.base import CBlock, Component, Context, ModelOutputThunk


class DummyBackend(Backend):
    """A backend for smoke testing."""

    def __init__(self, responses: list[str] | None):
        """Initializes the dummy backend, optionally with a list of dummy responses.

        Args:
            responses: If `None`, then the dummy backend always returns "dummy". Otherwise, returns the next item from responses. The generate function will throw an exception if a generate call is made after the list is exhausted.
        """
        self.responses = responses
        self.idx = 0

    def generate_from_context(
        self,
        action: Component | CBlock,
        ctx: Context,
        *,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> tuple[ModelOutputThunk, Context]:
        """See constructor for an exmplanation of how DummyBackends work."""
        assert format is None, "The DummyBackend does not support constrained decoding."
        if self.responses is None:
            mot = ModelOutputThunk(value="dummy")
            return mot, ctx.add(action).add(mot)
        elif self.idx < len(self.responses):
            return_value = ModelOutputThunk(value=self.responses[self.idx])
            self.idx += 1
            return return_value, ctx.add(action).add(return_value)
        else:
            raise Exception(
                f"DummyBackend expected no more than {len(self.responses)} calls."
            )
