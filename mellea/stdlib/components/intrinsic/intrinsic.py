"""Module for Intrinsics."""

from ....backends.adapters import AdapterType, fetch_intrinsic_metadata
from ....core import CBlock, Component, ModelOutputThunk, TemplateRepresentation


class Intrinsic(Component[str]):
    """A component representing an intrinsic."""

    def __init__(
        self, intrinsic_name: str, intrinsic_kwargs: dict | None = None
    ) -> None:
        """A component for rewriting messages using intrinsics.

        Intrinsics are special components that transform a chat completion request.
        These transformations typically take the form of:
        - parameter changes (typically structured outputs)
        - adding new messages to the chat
        - editing existing messages

        An intrinsic component should correspond to a loaded adapter.

        Args:
            intrinsic_name: the user-visible name of the intrinsic; must match a known
                name in Mellea's intrinsics catalog.
            intrinsic_kwargs: some intrinsics require kwargs when utilizing them;
                provide them here
        """
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
        """The set of all the constituent parts of the `Intrinsic`.

        Will need to be implemented by subclasses since not all intrinsics are output
        as text / messages.
        """
        return []  # TODO revisit this.

    def format_for_llm(self) -> TemplateRepresentation | str:
        """`Intrinsic` doesn't implement `format_for_default`.

        Formats the `Intrinsic` into a `TemplateRepresentation` or string.

        Returns: a `TemplateRepresentation` or string
        """
        raise NotImplementedError(
            "`Intrinsic` doesn't implement format_for_llm by default. You should only "
            "use an `Intrinsic` as the action and not as a part of the context."
        )

    def _parse(self, computed: ModelOutputThunk) -> str:
        """Parse the model output. Returns string value for now."""
        return computed.value if computed.value is not None else ""
