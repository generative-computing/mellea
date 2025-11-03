"""Module for Intrinsics."""

import pathlib
from copy import copy
from typing import cast

import granite_common

from mellea.backends.adapters.adapter import AdapterType
from mellea.stdlib.base import CBlock, Component, TemplateRepresentation


class Intrinsic(Component):
    """A component representing an intrinsic."""

    def __init__(
        self,
        intrinsic_name: str,
        intrinsic_kwargs: dict | None = None,
        adapter_types: list[AdapterType] = [AdapterType.ALORA, AdapterType.LORA],
    ) -> None:
        """A component for rewriting messages using intrinsics.

        Intrinsics are special components that transform a chat completion request.
        These transformations typically take the form of:
        - parameter changes (typically structured outputs)
        - adding new messages to the chat
        - editing existing messages

        An intrinsic component should correspond to a loaded adapter.

        Args:
            intrinsic_name: the name of the intrinsic; must match the adapter
            intrinsic_kwargs: some intrinsics require kwargs when utilizing them; provide them here
            adapter_types: list of adapter types that can be used for this intrinsic
        """
        self.intrinsic_name = intrinsic_name

        # Copy the list so that this intrinsic has its own list that can be modified independently.
        self.adapter_types = copy(adapter_types)

        if intrinsic_kwargs is None:
            intrinsic_kwargs = {}
        self.intrinsic_kwargs = intrinsic_kwargs

    def parts(self) -> list[Component | CBlock]:
        """The set of all the constituent parts of the `Intrinsic`.

        Will need to be implemented by subclasses since not all intrinsics are output
        as text / messages.
        """
        raise NotImplementedError("parts isn't implemented by default")

    def format_for_llm(self) -> TemplateRepresentation | str:
        """`Intrinsic` doesn't implement `format_for_default`. Formats the `Intrinsic` into a `TemplateRepresentation` or string.

        Returns: a `TemplateRepresentation` or string
        """
        raise NotImplementedError(
            "`Intrinsic` doesn't implement format_for_llm by default. You should only use an `Intrinsic` as the action and not as a part of the context."
        )
