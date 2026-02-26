"""Catalog of available intrinsics.

Catalog of intrinsics currently known to Mellea,including metadata about where to find
LoRA and aLoRA adapters that implement said intrinsics.
"""

import enum

import pydantic


class AdapterType(enum.Enum):
    """Possible types of adapters for a backend."""

    LORA = "lora"
    ALORA = "alora"


class IntriniscsCatalogEntry(pydantic.BaseModel):
    """A single row in the main intrinsics catalog table.

    We use Pydantic for this dataclass because the rest of Mellea also uses Pydantic.
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
    adapter_types: tuple[AdapterType, ...] = pydantic.Field(
        default=(AdapterType.LORA, AdapterType.ALORA),
        description="Adapter types that are known to be available for this intrinsic.",
    )


_RAG_REPO = "ibm-granite/granite-lib-rag-r1.0"
_CORE_REPO = "ibm-granite/rag-intrinsics-lib"
_CORE_R1_REPO = "ibm-granite/granite-lib-core-r1.0"


_INTRINSICS_CATALOG_ENTRIES = [
    ############################################
    # Core Intrinsics
    ############################################
    IntriniscsCatalogEntry(name="requirement_check", repo_id=_CORE_REPO),
    IntriniscsCatalogEntry(name="uncertainty", repo_id=_CORE_R1_REPO),
    ############################################
    # RAG Intrinsics
    ############################################
    IntriniscsCatalogEntry(
        name="answer_relevance_classifier",
        repo_id=_RAG_REPO,
        adapter_types=(AdapterType.LORA,),
    ),
    IntriniscsCatalogEntry(name="answer_relevance_rewriter", repo_id=_RAG_REPO),
    IntriniscsCatalogEntry(name="answerability", repo_id=_RAG_REPO),
    IntriniscsCatalogEntry(name="citations", repo_id=_RAG_REPO),
    IntriniscsCatalogEntry(name="context_relevance", repo_id=_RAG_REPO),
    IntriniscsCatalogEntry(name="hallucination_detection", repo_id=_RAG_REPO),
    IntriniscsCatalogEntry(
        name="query_clarification", repo_id=_RAG_REPO, adapter_types=(AdapterType.LORA,)
    ),
    IntriniscsCatalogEntry(name="query_rewrite", repo_id=_RAG_REPO),
]

_INTRINSICS_CATALOG = {e.name: e for e in _INTRINSICS_CATALOG_ENTRIES}
"""Catalog of intrinsics that Mellea knows about.

Mellea code should access this catalog via :func:`fetch_intrinsic_metadata()`"""


def known_intrinsic_names() -> list[str]:
    """:returns: List of all known user-visible names for intrinsics."""
    return list(_INTRINSICS_CATALOG.keys())


def fetch_intrinsic_metadata(intrinsic_name: str) -> IntriniscsCatalogEntry:
    """Retrieve information about the adapter that backs an intrinsic.

    :param intrinsic_name: User-visible name of the intrinsic

    :returns: Metadata about the adapter(s) that implement the intrinsic.
    """
    if intrinsic_name not in _INTRINSICS_CATALOG:
        raise ValueError(
            f"Unknown intrinsic name '{intrinsic_name}'. Valid names are "
            f"{known_intrinsic_names()}"
        )

    # Make a copy in case some naughty downstream code decides to modify the returned
    # value.
    return _INTRINSICS_CATALOG[intrinsic_name].model_copy()
