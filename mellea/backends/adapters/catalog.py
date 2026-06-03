"""Catalog of available intrinsics.

Catalog of intrinsics currently known to Mellea,including metadata about where to find
LoRA and aLoRA adapters that implement said intrinsics.
"""

import enum

import pydantic


def validate_revision(revision: str) -> str:
    """Validate a HuggingFace revision value.

    Accepts any non-empty string. HuggingFace's ``revision`` parameter takes a
    branch name, tag, or commit SHA; this validator mirrors that contract.
    Catalogue entries pin to commit SHAs by convention; that is enforced by
    review and (optionally) build-time drift checks rather than by this
    validator.

    Args:
        revision (str): Any non-empty string accepted by HuggingFace as a
            revision (branch name, tag, or commit SHA).

    Returns:
        str: The validated revision unchanged.

    Raises:
        ValueError: If `revision` is empty or whitespace-only.
    """
    if not revision.strip():
        raise ValueError("revision must be a non-empty string")
    return revision


class AdapterType(enum.Enum):
    """Possible types of adapters for a backend.

    Attributes:
        LORA (str): Standard LoRA adapter; value ``"lora"``.
        ALORA (str): Activated LoRA adapter; value ``"alora"``.
    """

    LORA = "lora"
    ALORA = "alora"


class IntriniscsCatalogEntry(pydantic.BaseModel):
    """A single row in the main intrinsics catalog table.

    We use Pydantic for this dataclass because the rest of Mellea also uses Pydantic.

    Attributes:
        name (str): User-visible name of the intrinsic.
        internal_name (str | None): Internal name used for adapter loading, or
            ``None`` if the same as ``name``.
        repo_id (str): HuggingFace repository where adapters for the intrinsic
            are located.
        revision (str): HuggingFace revision — branch name, tag, or commit SHA.
            Catalogue entries pin to commit SHAs by convention so loads are
            reproducible; the validator itself only requires a non-empty string.
            Note: this field is stored in the catalogue but not yet forwarded to
            the HuggingFace download call; wiring it through is deferred to a
            subsequent phase of the adapter-lifecycle epic (#929).
        adapter_types (tuple[AdapterType, ...]): Adapter types known to be
            available for this intrinsic; defaults to
            ``(AdapterType.LORA, AdapterType.ALORA)``.
    """

    name: str = pydantic.Field(description="User-visible name of the intrinsic.")
    internal_name: str | None = pydantic.Field(
        default=None,
        description="Internal name used for adapter loading, or None if the name used "
        "for that purpose is the same as self.name",
    )
    repo_id: str = pydantic.Field(
        description="Hugging Face repository (aka 'model') where adapters for the "
        "intrinsic are located."
    )
    revision: str = pydantic.Field(
        description="HuggingFace revision (branch name, tag, or commit SHA). "
        "Catalogue entries pin to commit SHAs by convention."
    )
    adapter_types: tuple[AdapterType, ...] = pydantic.Field(
        default=(AdapterType.LORA, AdapterType.ALORA),
        description="Adapter types that are known to be available for this intrinsic.",
    )

    @pydantic.field_validator("revision")
    @classmethod
    def _check_revision(cls, v: str) -> str:
        return validate_revision(v)


_RAG_REPO = "ibm-granite/granitelib-rag-r1.0"
_CORE_R1_REPO = "ibm-granite/granitelib-core-r1.0"
_GUARDIAN_REPO = "ibm-granite/granitelib-guardian-r1.0"

_RAG_SHA = "2f0b2c79c6731068625aca8045c2eb2e8912b353"  # main @ 2026-05-26
_CORE_R1_SHA = "d0a2a96a4cd07e96f0fe7ca29a42bfe088299d43"  # main @ 2026-05-26
_GUARDIAN_SHA = "773b254e98f993a605ec4b6259634906e0e64e8e"  # main @ 2026-05-26


_INTRINSICS_CATALOG_ENTRIES = [
    ############################################
    # Core Intrinsics
    ############################################
    IntriniscsCatalogEntry(
        name="context-attribution", repo_id=_CORE_R1_REPO, revision=_CORE_R1_SHA
    ),
    IntriniscsCatalogEntry(
        name="requirement-check", repo_id=_CORE_R1_REPO, revision=_CORE_R1_SHA
    ),
    IntriniscsCatalogEntry(
        name="uncertainty", repo_id=_CORE_R1_REPO, revision=_CORE_R1_SHA
    ),
    ############################################
    # RAG Intrinsics
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
    # Guardian Intrinsics
    ############################################
    IntriniscsCatalogEntry(
        name="policy-guardrails", repo_id=_GUARDIAN_REPO, revision=_GUARDIAN_SHA
    ),
    IntriniscsCatalogEntry(
        name="guardian-core", repo_id=_GUARDIAN_REPO, revision=_GUARDIAN_SHA
    ),
    IntriniscsCatalogEntry(
        name="factuality-detection", repo_id=_GUARDIAN_REPO, revision=_GUARDIAN_SHA
    ),
    IntriniscsCatalogEntry(
        name="factuality-correction", repo_id=_GUARDIAN_REPO, revision=_GUARDIAN_SHA
    ),
]

_INTRINSICS_CATALOG = {e.name: e for e in _INTRINSICS_CATALOG_ENTRIES}
"""Catalog of intrinsics that Mellea knows about.

Mellea code should access this catalog via :func:`fetch_intrinsic_metadata()`"""


def known_intrinsic_names() -> list[str]:
    """Return all known user-visible names for intrinsics.

    Returns:
        List of all known user-visible intrinsic names.
    """
    return list(_INTRINSICS_CATALOG.keys())


def fetch_intrinsic_metadata(intrinsic_name: str) -> IntriniscsCatalogEntry:
    """Retrieve information about the adapter that backs an intrinsic.

    Args:
        intrinsic_name (str): User-visible name of the intrinsic.

    Returns:
        IntriniscsCatalogEntry: Metadata about the adapter(s) that implement the
            intrinsic.

    Raises:
        ValueError: If ``intrinsic_name`` is not a known intrinsic name.
    """
    if intrinsic_name not in _INTRINSICS_CATALOG:
        raise ValueError(
            f"Unknown intrinsic name '{intrinsic_name}'. Valid names are "
            f"{known_intrinsic_names()}"
        )

    # Make a copy in case some naughty downstream code decides to modify the returned
    # value.
    return _INTRINSICS_CATALOG[intrinsic_name].model_copy()
