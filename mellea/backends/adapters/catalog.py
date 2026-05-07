"""Catalog of available intrinsics.

Catalog of intrinsics currently known to Mellea,including metadata about where to find
LoRA and aLoRA adapters that implement said intrinsics.
"""

import enum
import pathlib

import pydantic

# TODO(#1017): remove ``_OVERLAY_ROOT``, the ``io_yaml_overlay_dir`` catalog
# field, and the ``_overlays/`` directory once the Guardian intrinsics ship
# ``io.yaml`` upstream in the Granite Intrinsics Library. The overlays are a
# temporary bridge so the generic ``Intrinsic`` path works for Guardian; once
# upstream ships the templates, the HF-cached copies are authoritative.
_OVERLAY_ROOT = pathlib.Path(__file__).parent / "_overlays"


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
    adapter_types: tuple[AdapterType, ...] = pydantic.Field(
        default=(AdapterType.LORA, AdapterType.ALORA),
        description="Adapter types that are known to be available for this intrinsic.",
    )
    io_yaml_overlay_dir: pathlib.Path | None = pydantic.Field(
        default=None,
        description="Optional directory of in-repo ``io.yaml`` files that "
        "override the HuggingFace-cached versions. Layout mirrors the HF "
        "repo: ``<overlay_dir>/<canonical-model>/<lora|alora>/io.yaml``. "
        "Intended as a temporary bridge for intrinsics whose upstream "
        "io.yaml has not yet shipped; remove the entry once upstream "
        "ships the template.",
    )


# Mellea will update which repositories are linked as new ones come online. The original
# repos are on an older layout that will be changed.
_RAG_REPO = "ibm-granite/granitelib-rag-r1.0"
_CORE_REPO = "ibm-granite/rag-intrinsics-lib"  # Temporary; used by requirement checker
_CORE_R1_REPO = "ibm-granite/granitelib-core-r1.0"
_GUARDIAN_REPO = "ibm-granite/granitelib-guardian-r1.0"


_INTRINSICS_CATALOG_ENTRIES = [
    ############################################
    # Core Intrinsics
    ############################################
    IntriniscsCatalogEntry(name="context-attribution", repo_id=_CORE_R1_REPO),
    IntriniscsCatalogEntry(name="requirement-check", repo_id=_CORE_R1_REPO),
    IntriniscsCatalogEntry(
        name="requirement_check", repo_id=_CORE_REPO
    ),  # Necessary to support granite 3.2 and 3.3.
    IntriniscsCatalogEntry(name="uncertainty", repo_id=_CORE_R1_REPO),
    ############################################
    # RAG Intrinsics
    ############################################
    IntriniscsCatalogEntry(name="answerability", repo_id=_RAG_REPO),
    IntriniscsCatalogEntry(name="citations", repo_id=_RAG_REPO),
    IntriniscsCatalogEntry(name="context_relevance", repo_id=_RAG_REPO),
    IntriniscsCatalogEntry(name="hallucination_detection", repo_id=_RAG_REPO),
    IntriniscsCatalogEntry(name="query_clarification", repo_id=_RAG_REPO),
    IntriniscsCatalogEntry(name="query_rewrite", repo_id=_RAG_REPO),
    ############################################
    # Guardian Intrinsics
    ############################################
    IntriniscsCatalogEntry(
        name="policy-guardrails",
        repo_id=_GUARDIAN_REPO,
        io_yaml_overlay_dir=_OVERLAY_ROOT / "policy-guardrails",
    ),
    IntriniscsCatalogEntry(
        name="guardian-core",
        repo_id=_GUARDIAN_REPO,
        io_yaml_overlay_dir=_OVERLAY_ROOT / "guardian-core",
    ),
    IntriniscsCatalogEntry(
        name="factuality-detection",
        repo_id=_GUARDIAN_REPO,
        io_yaml_overlay_dir=_OVERLAY_ROOT / "factuality-detection",
    ),
    IntriniscsCatalogEntry(
        name="factuality-correction",
        repo_id=_GUARDIAN_REPO,
        io_yaml_overlay_dir=_OVERLAY_ROOT / "factuality-correction",
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
