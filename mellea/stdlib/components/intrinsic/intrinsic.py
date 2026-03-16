"""`Intrinsic` component for invoking fine-tuned adapter capabilities.

An `Intrinsic` component references a named adapter from Mellea's intrinsic catalog
and transforms a chat completion request — typically by injecting new messages,
modifying model parameters, or applying structured output constraints. It must be
paired with a backend that supports adapter loading (e.g. `LocalHFBackend` with an
attached `IntrinsicAdapter`).
"""

from ....backends.adapters import AdapterType, fetch_intrinsic_metadata
from ....core import CBlock, Component, ModelOutputThunk, TemplateRepresentation


class Intrinsic(Component[str]):
    """A component representing an intrinsic fine-tuned adapter capability.

    Intrinsics transform a chat completion request by injecting messages,
    modifying model parameters, or applying structured output constraints.
    Must be paired with a backend that supports adapter loading.

    Args:
        intrinsic_name (str): The user-visible name of the intrinsic; must match
            a known name in Mellea's intrinsics catalog.
        intrinsic_kwargs (dict | None): Optional keyword arguments required by
            the intrinsic at invocation time.

    Attributes:
        metadata: The intrinsic metadata fetched from the catalog.
        intrinsic_kwargs (dict): Keyword arguments passed to the intrinsic.
        intrinsic_name (str): User-visible name of this intrinsic (property).
        adapter_types (tuple[AdapterType, ...]): Available adapter types that
            implement this intrinsic (property).
    """

    def __init__(
        self, intrinsic_name: str, intrinsic_kwargs: dict | None = None
    ) -> None:
        """Initialize Intrinsic by fetching metadata for the named intrinsic from the catalog."""
        self.metadata = fetch_intrinsic_metadata(intrinsic_name)
        if intrinsic_kwargs is None:
            intrinsic_kwargs = {}
        self.intrinsic_kwargs = intrinsic_kwargs

    @property
    def intrinsic_name(self):
        """User-visible name of this intrinsic."""
        return self.metadata.name

    @property
    def adapter_types(self) -> tuple[AdapterType, ...]:
        """Tuple of available adapter types that implement this intrinsic."""
        return self.metadata.adapter_types

    def parts(self) -> list[Component | CBlock]:
        """Return the constituent parts of this intrinsic component.

        Will need to be implemented by subclasses since not all intrinsics
        produce text or message output.

        Returns:
            list[Component | CBlock]: Always an empty list for the base class.
        """
        return []  # TODO revisit this.

    def format_for_llm(self) -> TemplateRepresentation | str:
        """Not implemented for the base `Intrinsic` class.

        `Intrinsic` components are intended to be used as the *action* passed
        directly to the backend, not as a part of the context rendered by the
        formatter.

        Returns:
            TemplateRepresentation | str: Never returns; always raises.

        Raises:
            NotImplementedError: Always, because `Intrinsic` does not
                implement `format_for_llm` by default.
        """
        raise NotImplementedError(
            "`Intrinsic` doesn't implement format_for_llm by default. You should only "
            "use an `Intrinsic` as the action and not as a part of the context."
        )

    def _parse(self, computed: ModelOutputThunk) -> str:
        """Parse the model output. Returns string value for now."""
        return computed.value if computed.value is not None else ""
