# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""`Intrinsic` component for invoking fine-tuned adapter capabilities.

An `Intrinsic` component references a named adapter from Mellea's intrinsic catalog
and transforms a chat completion request — typically by injecting new messages,
modifying model parameters, or applying structured output constraints. It must be
paired with a backend that supports adapter loading (e.g. `LocalHFBackend` with a
composed `Adapter` — see `mellea.backends.adapters` — attached via `add_adapter`).
"""

from ....backends.adapters import AdapterType, fetch_intrinsic_metadata
from ....backends.adapters.catalog import IntrinsicsCatalogEntry
from ....core import Component, ModelOutputThunk, Span, TemplateRepresentation


class Intrinsic(Component[str]):
    """A component representing an intrinsic fine-tuned adapter capability.

    Intrinsics transform a chat completion request by injecting messages,
    modifying model parameters, or applying structured output constraints.
    Must be paired with a backend that supports adapter loading.

    Args:
        intrinsic_name (str): The user-visible name of the intrinsic. For a
            name outside Mellea's intrinsics catalog (a custom, non-catalog
            adapter function), `adapter_types` must be provided explicitly —
            this is what lets a custom `Intrinsic` construct without a
            catalog entry, replacing the deprecated `CustomIntrinsicAdapter`
            shim's catalog monkey-patch (Epic #929, issue #1144).
        intrinsic_kwargs (dict | None): Optional keyword arguments required by
            the intrinsic at invocation time.
        adapter_types (tuple[AdapterType, ...] | None): Adapter types this
            intrinsic supports. Required for a name outside the catalog;
            optional (defaults to the catalog entry's own types) otherwise.

    Attributes:
        metadata: The intrinsic metadata fetched from the catalog, or a
            synthetic entry for a name outside it (see `adapter_types` above).
        intrinsic_kwargs (dict): Keyword arguments passed to the intrinsic.
        intrinsic_name (str): User-visible name of this intrinsic (property).
        adapter_types (tuple[AdapterType, ...]): Available adapter types that
            implement this intrinsic (property). Defaults to values in self.metadata.

    Raises:
        ValueError: `intrinsic_name` is not in the catalog and `adapter_types`
            was not provided — there is no way to know which adapter
            realities to look up without either.
    """

    def __init__(
        self,
        intrinsic_name: str,
        intrinsic_kwargs: dict | None = None,
        adapter_types: tuple[AdapterType, ...] | None = None,
    ) -> None:
        """Initialize Intrinsic by fetching metadata for the named intrinsic from the catalog."""
        try:
            self.metadata = fetch_intrinsic_metadata(intrinsic_name)
        except ValueError:
            if adapter_types is None:
                raise
            # A custom, non-catalog adapter function: the caller has already
            # told us which adapter types to look up, so a catalog entry
            # (whose repo_id/revision this class never reads — see
            # `adapter_types` and `intrinsic_name` properties below) isn't
            # needed to construct a usable Intrinsic. "not-a-catalog-entry"
            # (rather than "unpinned") avoids colliding with the
            # `mellea.adapter_function.*` telemetry sentinel, which already
            # uses the string "unpinned" to mean "revision is None" for a
            # real, catalog-resolvable adapter — a different situation from
            # this synthetic entry having no revision concept at all.
            self.metadata = IntrinsicsCatalogEntry(
                name=intrinsic_name,
                repo_id="",
                revision="not-a-catalog-entry",
                adapter_types=adapter_types,
            )
        if intrinsic_kwargs is None:
            intrinsic_kwargs = {}
        self.intrinsic_kwargs = intrinsic_kwargs
        self._adapter_types = adapter_types

    @property
    def intrinsic_name(self):
        """User-visible name of this intrinsic."""
        return self.metadata.name

    @property
    def adapter_types(self) -> tuple[AdapterType, ...]:
        """Tuple of available adapter types that implement this intrinsic."""
        return (
            self._adapter_types
            if self._adapter_types is not None
            else self.metadata.adapter_types
        )

    def parts(self) -> list[Span]:
        """Return the constituent parts of this intrinsic component.

        Will need to be implemented by subclasses since not all intrinsics
        produce text or message output.

        Returns:
            list[Span]: Always an empty list for the base class.
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
