"""Catalog of available adapter functions.

Catalog of adapter functions currently known to Mellea, including metadata about where
to find LoRA and aLoRA adapters that implement them.
"""

import enum

import pydantic


def validate_revision(revision: str) -> str:
    """Validate a Hugging Face revision value.

    Accepts any non-empty string. Hugging Face's ``revision`` parameter takes a
    branch name, tag, or commit SHA; this validator mirrors that contract.
    Catalogue entries pin to commit SHAs by convention; that is enforced by
    review and (optionally) build-time drift checks rather than by this
    validator.

    Args:
        revision (str): Any non-empty string accepted by Hugging Face as a
            revision (branch name, tag, or commit SHA).

    Returns:
        str: The validated revision unchanged.

    Raises:
        ValueError: If `revision` is empty, whitespace-only, or has leading
            or trailing whitespace.
    """
    if not revision.strip():
        raise ValueError("revision must be a non-empty string")
    if revision != revision.strip():
        raise ValueError("revision must not have leading or trailing whitespace")
    return revision


class AdapterType(enum.Enum):
    """Possible types of adapters for a backend.

    Attributes:
        LORA (str): Standard LoRA adapter; value ``"lora"``.
        ALORA (str): aLoRA adapter (shares model KV cache across adapter functions); value ``"alora"``.
    """

    LORA = "lora"
    ALORA = "alora"


class IntriniscsCatalogEntry(pydantic.BaseModel):
    """A single row in the main adapter function catalog table.

    We use Pydantic for this dataclass because the rest of Mellea also uses Pydantic.

    Attributes:
        name (str): User-visible name of the adapter function. May contain hyphens when
            that matches the upstream adapter name; prefer ``effective_capability``
            to form the stable capability token. Must be non-empty with no leading
            or trailing whitespace.
        capability (str | None): Stable Mellea-level capability token, independent
            of the upstream adapter name. Uses underscores. When ``None``,
            :attr:`effective_capability` falls back to ``name``. When set, must be
            non-empty with no leading or trailing whitespace.
        internal_name (str | None): Internal name used for adapter loading, or
            ``None`` if the same as ``name``.
        repo_id (str): Hugging Face repository where adapters for the adapter function
            are located.
        revision (str): Hugging Face revision — branch name, tag, or commit SHA.
            Catalogue entries pin to commit SHAs by convention so loads are
            reproducible; the validator itself only requires a non-empty string.
            Note: this field is stored in the catalogue but not yet forwarded to
            the Hugging Face download call; wiring it through is deferred to a
            subsequent phase of the adapter-lifecycle epic (#929).
        adapter_types (tuple[AdapterType, ...]): Adapter types known to be
            available for this adapter function; defaults to
            ``(AdapterType.LORA, AdapterType.ALORA)``.
    """

    name: str = pydantic.Field(
        description="User-visible name of the adapter function. Non-empty, no leading/trailing whitespace."
    )
    capability: str | None = pydantic.Field(
        default=None,
        description=(
            "Stable capability token, independent of the upstream adapter name. "
            "Uses underscores; no leading/trailing whitespace. When ``None``, "
            "``effective_capability`` falls back to ``name``."
        ),
    )
    internal_name: str | None = pydantic.Field(
        default=None,
        description="Internal name used for adapter loading, or None if the name used "
        "for that purpose is the same as self.name",
    )
    repo_id: str = pydantic.Field(
        description="Hugging Face repository (aka 'model') where adapters for the "
        "adapter function are located."
    )
    revision: str = pydantic.Field(
        description="Hugging Face revision (branch name, tag, or commit SHA). "
        "Catalogue entries pin to commit SHAs by convention."
    )
    adapter_types: tuple[AdapterType, ...] = pydantic.Field(
        default=(AdapterType.LORA, AdapterType.ALORA),
        description="Adapter types that are known to be available for this adapter function.",
    )

    @pydantic.field_validator("name")
    @classmethod
    def _check_name(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("name must be a non-empty string")
        if v != v.strip():
            raise ValueError("name must not have leading or trailing whitespace")
        return v

    @pydantic.field_validator("capability")
    @classmethod
    def _check_capability(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if not v.strip():
            raise ValueError(
                "capability must be a non-empty string when set; use None to fall back to name"
            )
        if v != v.strip():
            raise ValueError("capability must not have leading or trailing whitespace")
        return v

    @pydantic.field_validator("revision")
    @classmethod
    def _check_revision(cls, v: str) -> str:
        return validate_revision(v)

    @property
    def effective_capability(self) -> str:
        """Return the stable capability token for this adapter function.

        Returns ``capability`` when explicitly set; falls back to ``name``
        otherwise. Use this property — not ``name`` — whenever building the
        capability vocabulary or resolving an
        :class:`~mellea.backends.adapters.Identity` capability.

        Returns:
            str: Capability token, guaranteed non-empty.
        """
        return self.capability if self.capability is not None else self.name


_RAG_REPO = "ibm-granite/granitelib-rag-r1.0"
_CORE_R1_REPO = "ibm-granite/granitelib-core-r1.0"
_GUARDIAN_REPO = "ibm-granite/granitelib-guardian-r1.0"

_RAG_SHA = "2f0b2c79c6731068625aca8045c2eb2e8912b353"  # main @ 2026-05-26
_CORE_R1_SHA = "d0a2a96a4cd07e96f0fe7ca29a42bfe088299d43"  # main @ 2026-05-26
_GUARDIAN_SHA = "773b254e98f993a605ec4b6259634906e0e64e8e"  # main @ 2026-05-26


_INTRINSICS_CATALOG_ENTRIES = [
    ############################################
    # Core adapter functions
    ############################################
    IntriniscsCatalogEntry(
        name="context-attribution",
        capability="context_attribution",
        repo_id=_CORE_R1_REPO,
        revision=_CORE_R1_SHA,
    ),
    IntriniscsCatalogEntry(
        name="requirement-check",
        capability="requirement_check",
        repo_id=_CORE_R1_REPO,
        revision=_CORE_R1_SHA,
    ),
    IntriniscsCatalogEntry(
        name="uncertainty", repo_id=_CORE_R1_REPO, revision=_CORE_R1_SHA
    ),
    ############################################
    # RAG adapter functions
    ############################################
    IntriniscsCatalogEntry(name="answerability", repo_id=_RAG_REPO, revision=_RAG_SHA),
    IntriniscsCatalogEntry(name="citations", repo_id=_RAG_REPO, revision=_RAG_SHA),
    IntriniscsCatalogEntry(
        name="context_relevance", repo_id=_RAG_REPO, revision=_RAG_SHA
    ),
    IntriniscsCatalogEntry(
        name="hallucination_detection", repo_id=_RAG_REPO, revision=_RAG_SHA
    ),
    IntriniscsCatalogEntry(
        name="query_clarification", repo_id=_RAG_REPO, revision=_RAG_SHA
    ),
    IntriniscsCatalogEntry(name="query_rewrite", repo_id=_RAG_REPO, revision=_RAG_SHA),
    ############################################
    # Guardian adapter functions
    ############################################
    IntriniscsCatalogEntry(
        name="policy-guardrails",
        capability="policy_guardrails",
        repo_id=_GUARDIAN_REPO,
        revision=_GUARDIAN_SHA,
    ),
    IntriniscsCatalogEntry(
        name="guardian-core",
        capability="guardian_core",
        repo_id=_GUARDIAN_REPO,
        revision=_GUARDIAN_SHA,
    ),
    IntriniscsCatalogEntry(
        name="factuality-detection",
        capability="factuality_detection",
        repo_id=_GUARDIAN_REPO,
        revision=_GUARDIAN_SHA,
    ),
    IntriniscsCatalogEntry(
        name="factuality-correction",
        capability="factuality_correction",
        repo_id=_GUARDIAN_REPO,
        revision=_GUARDIAN_SHA,
    ),
]

_INTRINSICS_CATALOG = {e.name: e for e in _INTRINSICS_CATALOG_ENTRIES}
"""Catalog of adapter functions that Mellea knows about.

Mellea code should access this catalog via :func:`fetch_intrinsic_metadata()`"""

_effective_caps = [e.effective_capability for e in _INTRINSICS_CATALOG_ENTRIES]
if len(_effective_caps) != len(set(_effective_caps)):
    _dupes = {c for c in _effective_caps if _effective_caps.count(c) > 1}
    raise ValueError(
        f"Duplicate effective_capability values in intrinsics catalog: {_dupes}"
    )
del _effective_caps


def known_intrinsic_names() -> list[str]:
    """Return all known user-visible names for adapter functions.

    Returns:
        List of all known user-visible adapter function names.
    """
    return list(_INTRINSICS_CATALOG.keys())


def fetch_intrinsic_metadata(intrinsic_name: str) -> IntriniscsCatalogEntry:
    """Retrieve catalog metadata for the adapter that implements an adapter function.

    Args:
        intrinsic_name (str): User-visible name of the adapter function.

    Returns:
        IntriniscsCatalogEntry: Metadata about the adapter(s) that implement the
            adapter function.

    Raises:
        ValueError: If ``intrinsic_name`` is not a known adapter function name.
    """
    if intrinsic_name not in _INTRINSICS_CATALOG:
        raise ValueError(
            f"Unknown intrinsic name '{intrinsic_name}'. Valid names are "
            f"{known_intrinsic_names()}"
        )

    # Make a copy in case some naughty downstream code decides to modify the returned
    # value.
    return _INTRINSICS_CATALOG[intrinsic_name].model_copy()
