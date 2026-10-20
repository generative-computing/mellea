# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapter classes for adding fine-tuned modules to inference backends.

The primary public surface is :func:`AdapterMixin.resolve_adapter` (find or lazily
register an adapter by capability name) and :meth:`AdapterMixin._find_adapter`
(look up a registered adapter).  :class:`AdapterMixin` is mixed into backends that
support runtime adapter loading and unloading.

`LocalHFAdapter`, `IntrinsicAdapter`, and `EmbeddedIntrinsicAdapter` are
**deprecation shims** retained for backwards compatibility.  They satisfy
`isinstance(x, _core.Adapter)` but delegate all behaviour to the new dataclass.
`get_adapter_for_intrinsic` is similarly deprecated; prefer `resolve_adapter`.
"""

import abc
import contextlib
import copy
import hashlib
import pathlib
import re
import shutil
import tempfile
import time
import warnings
from collections.abc import Callable, Sequence
from typing import Any, ClassVar, Literal, TypeAlias, TypeVar, cast

import yaml

from ...core import Backend, MelleaLogger
from ...formatters.granite import intrinsics as intrinsics
from ...helpers.event_loop_helper import _run_async_in_thread
from ...plugins.manager import has_plugins, invoke_hook
from ...plugins.types import HookType
from ..model_options import ModelOption
from ._core import (
    Adapter as _AdapterCore,
    AdapterSchemaMismatchError,
    EmbeddedBinding,
    Identity,
    LocalFileBinding,
    WeightsBinding,
)
from .capabilities import DECISION_CAPABILITY
from .catalog import AdapterType, fetch_intrinsic_metadata, known_intrinsic_names
from .io_contracts import get_io_contract


class Adapter(abc.ABC):
    """An adapter that can be added to a single backend.

    An adapter can only be registered with one backend at a time. Use
    `adapter.qualified_name` when referencing the adapter after adding it.

    Args:
        name (str): Human-readable name of the adapter.
        adapter_type (AdapterType): Enum describing the adapter type (e.g.
            `AdapterType.LORA` or `AdapterType.ALORA`).

    Attributes:
        qualified_name (str): Unique name used for loading and lookup; formed
            as `"<name>_<adapter_type.value>"`.
        backend (Backend | None): The backend this adapter has been added to,
            or `None` if not yet added.
        path (str | None): Filesystem path to the adapter weights; set when
            the adapter is added to a backend.
    """

    def __init__(self, name: str, adapter_type: AdapterType):
        """Initialize Adapter with a name and adapter type."""
        self.name = name
        self.adapter_type = adapter_type
        self.qualified_name = name + "_" + adapter_type.value
        """the name of the adapter to use when loading / looking it up"""

        self.backend: Backend | None = None
        """set when the adapter is added to a backend"""

        self.path: str | None = None
        """set when the adapter is added to a backend"""


class LocalHFAdapter(Adapter):
    """Abstract adapter subclass for locally loaded Hugging Face model backends.

    Subclasses must implement `get_local_hf_path` to return the filesystem path
    from which adapter weights should be loaded given a base model name.
    """

    @abc.abstractmethod
    def get_local_hf_path(self, base_model_name: str) -> str:
        """Return the local filesystem path from which adapter weights should be loaded.

        Args:
            base_model_name (str): The base model name; typically the last component
                of the Hugging Face model ID (e.g. `"granite-4.0-micro"`).

        Returns:
            str: Filesystem path to the adapter weights directory.
        """
        ...


class _ShimWeightsBinding(WeightsBinding):
    """Placeholder weights binding for the deprecated IntrinsicAdapter shims.

    All lifecycle verbs raise NotImplementedError; it exists only so the
    shims can satisfy the Adapter protocol.
    """

    def prepare(self) -> None:
        raise NotImplementedError("WeightsBinding not yet implemented")

    def activate(self) -> None:
        raise NotImplementedError("WeightsBinding not yet implemented")

    def deactivate(self) -> None:
        raise NotImplementedError("WeightsBinding not yet implemented")

    def release(self) -> None:
        raise NotImplementedError("WeightsBinding not yet implemented")


class IntrinsicAdapter(LocalHFAdapter, _AdapterCore):
    """Deprecated shim for adapters that implement adapter functions.

    Deprecated:
        Use :class:`~mellea.backends.adapters.Adapter` directly.
        `IntrinsicAdapter` will be removed in a future release (Epic #929,
        issue #1144).

    Subtype of :class:`Adapter` for models that:

    * implement adapter functions
    * are packaged as LoRA or aLoRA adapters on top of a base model
    * use the shared model loading code in `mellea.formatters.granite.intrinsics`
    * use the shared input and output processing code in
      `mellea.formatters.granite.intrinsics`

    Args:
        intrinsic_name (str): Name of the adapter function (e.g. `"answerability"`);
            the adapter's `qualified_name` will be derived from this.
        adapter_type (AdapterType): Enum describing the adapter type; defaults to
            `AdapterType.ALORA`.
        config_file (str | pathlib.Path | None): Path to a YAML config file defining
            the adapter function's I/O transformations; mutually exclusive with
            `config_dict`.
        config_dict (dict | None): Dict defining the adapter function's I/O
            transformations; mutually exclusive with `config_file`.
        base_model_name (str | None): Base model name used to look up the I/O
            processing config when neither `config_file` nor `config_dict` are
            provided.

    Attributes:
        intrinsic_name (str): Name of the adapter function this adapter implements.
        intrinsic_metadata (IntrinsicsCatalogEntry): Catalog metadata for the adapter function.
        base_model_name (str | None): Base model name provided at construction, if any.
        adapter_type (AdapterType): The adapter type (`LORA` or `ALORA`).
        config (dict): Parsed I/O transformation configuration for the adapter function.

    Note:
        `identity`, `io_contract`, and `weights` are internal scaffolding populated
        in `__init__` to satisfy the :class:`~mellea.backends.adapters.Adapter`
        protocol; they are not meaningful consumer-facing attributes. `io_contract`
        is the real, declared contract for `intrinsic_name` (issue #1516); `weights`
        remains the Phase 1 `_ShimWeightsBinding` placeholder and raises
        `NotImplementedError` until Phase 2 (issue #1141) replaces it.
    """

    def __setattr__(self, name: str, value: object) -> None:
        """Allow mutation; bypasses the frozen restriction on _AdapterCore."""
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        """Allow deletion; bypasses the frozen restriction on _AdapterCore."""
        object.__delattr__(self, name)

    def __init__(
        self,
        intrinsic_name: str,
        adapter_type: AdapterType = AdapterType.ALORA,
        config_file: str | pathlib.Path | None = None,
        config_dict: dict | None = None,
        base_model_name: str | None = None,
    ):
        """Initialize IntrinsicAdapter for the named adapter function, loading its I/O configuration."""
        warnings.warn(
            "IntrinsicAdapter is deprecated; use Adapter directly (Epic #929, issue #1144).",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(intrinsic_name, adapter_type)

        self.intrinsic_name = intrinsic_name
        self.intrinsic_metadata = fetch_intrinsic_metadata(intrinsic_name)
        self.base_model_name = base_model_name

        if adapter_type not in self.intrinsic_metadata.adapter_types:
            raise ValueError(
                f"Adapter function '{intrinsic_name}' not available as an adapter of type "
                f"'{adapter_type}. Available types are "
                f"{self.intrinsic_metadata.adapter_types}."
            )
        self.adapter_type = adapter_type

        # If any of the optional params are specified, attempt to set up the
        # config for the adapter function here.
        if config_file and config_dict:
            raise ValueError(
                f"Conflicting values for config_file and config_dict "
                f"parameters provided. Values were {config_file=} "
                f"and {config_dict=}"
            )
        if config_file is None and config_dict is None and self.base_model_name is None:
            raise ValueError(
                "At least one of [config_file, config_dict, base_model_name] "
                "must be provided."
            )
        if config_file is None and config_dict is None:
            assert self.base_model_name is not None, (
                "must provide `base_model_name` if not providing a `config_file` or `config_dict`"
            )
            # We're converting the adapter type to a boolean flag here.
            assert adapter_type in (AdapterType.ALORA, AdapterType.LORA), (
                f"{adapter_type} not supported"
            )
            is_alora = self.adapter_type == AdapterType.ALORA
            config_file = intrinsics.obtain_io_yaml(
                self.intrinsic_name,
                self.base_model_name,
                self.intrinsic_metadata.repo_id,
                revision=self.intrinsic_metadata.revision,
                alora=is_alora,
            )
        if config_file:
            with open(config_file, encoding="utf-8") as f:
                config_dict = yaml.safe_load(f)
                if not isinstance(config_dict, dict):
                    raise ValueError(
                        f"YAML file {config_file} does not evaluate to a "
                        f"dictionary when parsed."
                    )
        assert config_dict is not None  # Code above should initialize this variable
        self.config: dict = config_dict

        # Populate the new Adapter triple so isinstance(self, _AdapterCore) holds.
        # io_contract comes from the same registry resolve_adapter() consults
        # (see issue #1516), not a placeholder. weights stays the Phase 2
        # _ShimWeightsBinding placeholder; that axis is #1141/#1142.
        _AdapterCore.__init__(
            self,
            identity=Identity(
                name=intrinsic_name,
                adapter_type="alora"
                if self.adapter_type == AdapterType.ALORA
                else "lora",
                capability=self.intrinsic_metadata.effective_capability,
            ),
            io_contract=get_io_contract(intrinsic_name),
            weights=_ShimWeightsBinding(),
        )

    def get_local_hf_path(self, base_model_name: str) -> str:
        """Return the local filesystem path from which adapter weights should be loaded.

        Downloads the adapter weights if they are not already cached locally.

        Args:
            base_model_name (str): The base model name; typically the last component
                of the Hugging Face model ID (e.g. `"granite-3.3-8b-instruct"`).

        Returns:
            str: Filesystem path to the downloaded adapter weights directory.
        """
        return self.download_and_get_path(base_model_name)

    def download_and_get_path(self, base_model_name: str) -> str:
        """Download the required adapter function files if necessary and return the path to them.

        Args:
            base_model_name: the base model; typically the last part of the Hugging Face
                model id like "granite-3.3-8b-instruct"

        Returns:
            a path to the files
        """
        is_alora = self.adapter_type == AdapterType.ALORA
        return str(
            intrinsics.obtain_lora(
                self.intrinsic_name,
                base_model_name,
                self.intrinsic_metadata.repo_id,
                revision=self.intrinsic_metadata.revision,
                alora=is_alora,
            )
        )


T = TypeVar("T")


def _composed_adapter_key(adapter: "_AdapterCore") -> str:
    """Return the registry key for a composed `Adapter`, mirroring `qualified_name`.

    A composed `Adapter` (unlike the deprecated shims) has no `qualified_name`
    of its own; backends key their registries on this instead. For a
    LocalFile/PEFT composed adapter, this must produce the same string as
    `adapter.weights.qualified_name` (`LocalFileBinding`'s own key) so a
    backend's `_added_adapters` (keyed on the binding) and `_composed_adapters`
    (keyed on this function) agree; which holds because `Identity.adapter_type`
    is the `Literal["lora", "alora"]` string and `AdapterType.LORA.value`/
    `AdapterType.ALORA.value` are those same two strings. `LocalHFBackend.add_adapter`
    enforces this at registration time for a `LocalFileBinding` (see the
    `NOTE(#1516)` on the `Adapter` dataclass); a composed `Adapter` whose
    `identity.adapter_type` disagrees with its own `weights.adapter_type`
    otherwise produces two different keys instead of one.

    Args:
        adapter: The composed adapter to key.

    Returns:
        str: `"<identity.name>_<identity.adapter_type>"`.
    """
    return f"{adapter.identity.name}_{adapter.identity.adapter_type}"


_DECIDER_CONFIG_FIELDS = (
    "system_prompt",
    "state_template",
    "invocation_text",
    "instruction",
    "transformations",
)
"""Decider fields of an io.yaml, merged field-by-field with the carried config."""

_CARRIED_DECIDER_PROVENANCE = (
    "mellea-carried interim config for the decision capability, derived from "
    "the model team's reference decider implementation (2026-10-15); "
    "applies only while the published io.yaml omits the decider fields and "
    "retires silently once it carries them"
)
"""Provenance stamp for the carried decider config."""

_CARRIED_DECIDER_CONFIG: dict = {
    "system_prompt": (
        "You are a decision engine. Read the state, then answer the question "
        "by replying with exactly one option label from the list. Do not "
        "explain."
    ),
    "state_template": "State:\n{state}",
    "invocation_text": "### DECISION",
    "instruction": (
        "### DECISION\nQuestion:\n{question}\n\nOptions:\n{options_block}\n\n"
        "Reply with exactly one option label."
    ),
    "transformations": [
        {"type": "first_token", "labels": list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")}
    ],
    "parameters": {
        "max_completion_tokens": 1,
        "temperature": 0.0,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    },
}
"""Interim augmented io.yaml content for the decision capability.

The published io.yaml of the decision capability is near-empty (decider
fields present-with-null), so mellea carries the trained prompt layout and
one-token readout contract until the model team publishes them. Applied at
one named point; `_discover_embedded_adapters`, which every composed
registration path traverses; with guardrails:

- field-by-field: for each decider field, a null published value gets the
  carried one, an equal published value is used silently, and a different
  published value wins with a WARNING naming both;
- `parameters` merge per-key, published winning;
- `response_format` is forced null for `first_token` capabilities (a
  non-null value would impose guided decoding on a one-token answer);
- provenance is stamped under the `provenance` key, naming the fields
  that came from the carried config;
- the merged config is validated (required placeholders, first_token rule);
  if invalid, the decision capability is not registered and the reason is
  surfaced through `resolve_adapter`'s error.

The golden-layout unit test (env-gated on staged artefacts) checks
message-list parity of the rendered layout with the reference
construction; it is not a numerical-parity or CI check."""


def _decider_field_equal(published: object, carried: object) -> bool:
    """Compare a decider field for equality."""
    return published == carried


def _validate_merged_decider_config(config: dict) -> str | None:
    """Validate a merged decider config; return the failure reason or None.

    The config is valid only if every field the rewriter and readout depend
    on is present and well-formed: a string `system_prompt`, a
    `state_template` containing `{state}`, an `instruction` containing
    `{question}`, `{options_block}`, and the `invocation_text` exactly once,
    and a `transformations` list with a `first_token` rule carrying labels.
    """
    system_prompt = config.get("system_prompt")
    if not isinstance(system_prompt, str) or not system_prompt:
        return (
            "merged decider config has no string system_prompt "
            f"(got {type(system_prompt).__name__})"
        )
    state_template = config.get("state_template")
    if not isinstance(state_template, str) or "{state}" not in state_template:
        return (
            "merged decider config state_template is missing the "
            f"{{state}} placeholder (got {state_template!r:.80})"
        )
    invocation_text = config.get("invocation_text")
    instruction = config.get("instruction")
    if not isinstance(instruction, str):
        return (
            "merged decider config has no string instruction "
            f"(got {type(instruction).__name__})"
        )
    for placeholder in ("{question}", "{options_block}"):
        if placeholder not in instruction:
            return (
                f"merged decider config instruction is missing the "
                f"{placeholder} placeholder"
            )
    if not isinstance(invocation_text, str) or instruction.count(invocation_text) != 1:
        return (
            f"merged decider config instruction must contain invocation_text "
            f"{invocation_text!r} exactly once (found "
            f"{instruction.count(invocation_text) if isinstance(invocation_text, str) else 0})"
        )
    transformations = config.get("transformations")
    if not isinstance(transformations, list) or not any(
        isinstance(t, dict)
        and t.get("type") == "first_token"
        and isinstance(t.get("labels"), list)
        and t["labels"]
        for t in transformations
    ):
        return (
            "merged decider config transformations must include a first_token "
            "rule with a non-empty labels list"
        )
    return None


def _check_decision_template_invocation(
    source: str,
    *,
    revision: str = "main",
    cache_dir: str | None = None,
    merged_invocation_text: str | None,
) -> None:
    """Log a WARNING when the checkpoint's chat template disagrees with the merged config.

    Parses the template's `adapter_map` (never renders it: the template is
    untrusted Hub content) and compares the `decision` entry's
    `invocation_text` against the merged decider config's, and its `token`
    against `adapter_index.json`'s `control_token.token`. Any mismatch,
    missing template, or parse failure logs a WARNING; the decision
    capability registers regardless, because a false positive that disabled
    registration would be worse than the mismatch it reports.

    Args:
        source: Local model directory or Hub repo ID, as passed to discovery.
        revision: Git revision (only used for Hub lookups).
        cache_dir: Cache directory (only used for Hub lookups).
        merged_invocation_text: The `invocation_text` from the merged decider
            config, or None when the merged config does not carry one.
    """
    logger = MelleaLogger.get_logger()
    if merged_invocation_text is None:
        return

    template_path: pathlib.Path | None = None
    index_path: pathlib.Path | None = None
    if pathlib.Path(source).is_dir():
        template_path = pathlib.Path(source) / "chat_template.jinja"
        index_path = pathlib.Path(source) / "adapter_index.json"
    else:
        try:
            import huggingface_hub

            template_path = pathlib.Path(
                huggingface_hub.hf_hub_download(
                    repo_id=source,
                    filename="chat_template.jinja",
                    revision=revision,
                    cache_dir=cache_dir,
                )
            )
            index_path = pathlib.Path(
                huggingface_hub.hf_hub_download(
                    repo_id=source,
                    filename="adapter_index.json",
                    revision=revision,
                    cache_dir=cache_dir,
                )
            )
        except Exception:
            logger.warning(
                "Could not verify the decision adapter's invocation marker: "
                "failed to fetch chat_template.jinja or adapter_index.json "
                "from %r at revision %r.",
                source,
                revision,
                exc_info=True,
            )
            template_path = None
            index_path = None

    if template_path is None or not template_path.is_file():
        logger.warning(
            "Could not verify the decision adapter's invocation marker: no "
            "chat_template.jinja found at %r. If the served template's "
            "invocation text differs from %r, decision calls will silently "
            "run on the base model.",
            source,
            merged_invocation_text,
        )
        return

    try:
        import jinja2

        template_src = template_path.read_text(encoding="utf-8")
        ast = jinja2.Environment().parse(template_src)
        adapter_map = next(
            (
                n.node.as_const()  # type: ignore[attr-defined]  # Assign.node is Expr at runtime; stubs say Node
                for n in ast.find_all(jinja2.nodes.Assign)
                if isinstance(n.target, jinja2.nodes.Name)
                and n.target.name == "adapter_map"
            ),
            None,
        )
        if not isinstance(adapter_map, dict):
            logger.warning(
                "Could not verify the decision adapter's invocation marker: "
                "chat_template.jinja at %r has no adapter_map assignment.",
                source,
            )
            return

        decision_entry = adapter_map.get("decision")
        if not isinstance(decision_entry, dict):
            logger.warning(
                "Could not verify the decision adapter's invocation marker: "
                "chat_template.jinja at %r has no 'decision' entry in its "
                "adapter_map (entries: %s).",
                source,
                sorted(adapter_map),
            )
            return

        template_invocation = decision_entry.get("invocation_text")
        if template_invocation != merged_invocation_text:
            logger.warning(
                "The checkpoint's chat template activates the decision "
                "adapter at %r, but mellea's merged config uses %r. Every "
                "decision call will silently run on the base model. Pin the "
                "checkpoint and mellea to the same revision, or report the "
                "mismatch.",
                template_invocation,
                merged_invocation_text,
            )

        if index_path is not None and index_path.is_file():
            import json as _json

            with open(index_path, encoding="utf-8") as f:
                index = _json.load(f)
            for entry in index.get("adapters", []):
                if entry.get("adapter_name") == "decision":
                    index_token = (entry.get("control_token") or {}).get("token")
                    template_token = decision_entry.get("token")
                    if index_token != template_token:
                        logger.warning(
                            "The checkpoint's chat template names control "
                            "token %r for the decision adapter, but "
                            "adapter_index.json declares %r.",
                            template_token,
                            index_token,
                        )
                    break
    except Exception:
        logger.warning(
            "Could not verify the decision adapter's invocation marker: "
            "failed to parse chat_template.jinja at %r.",
            source,
            exc_info=True,
        )


def _apply_carried_decider_config(
    name: str, config: dict
) -> tuple[dict | None, str | None]:
    """Merge the carried decider config into a discovered io.yaml config, field by field.

    For each decider field: a null published value gets the carried one, an
    equal published value is used silently, and a different published value
    wins with a WARNING naming both. The merged config is then validated;
    if invalid, returns (None, reason) so the caller can skip the
    capability and surface the specific reason to the caller's error.

    Args:
        name (str): Adapter function name from the model's adapter index.
        config (dict): Parsed io.yaml configuration for that adapter.

    Returns:
        tuple[dict | None, str | None]: The configuration to register and
        None, or None and the skip reason when the merged config is
        invalid. The input config and None, unchanged, for any capability
        other than the decision capability.
    """
    if name != DECISION_CAPABILITY:
        return config, None

    logger = MelleaLogger.get_logger()
    merged = dict(config)
    # Deep-copy the carried fields: the merged config travels to backends
    # and rewriters as per-adapter state, and sharing mutable structure
    # with the module-level constant would let one adapter's consumers
    # corrupt every other's.
    carried = copy.deepcopy(_CARRIED_DECIDER_CONFIG)
    carried_fields_used: list[str] = []
    for field in _DECIDER_CONFIG_FIELDS:
        published_value = config.get(field)
        if published_value is None:
            merged[field] = carried[field]
            carried_fields_used.append(field)
        elif not _decider_field_equal(published_value, carried[field]):
            logger.warning(
                "Published io.yaml for adapter function '%s' field '%s' "
                "differs from mellea's carried decider config; the published "
                "value wins. Published: %r; carried: %r.",
                name,
                field,
                published_value,
                carried[field],
            )

    published_parameters = config.get("parameters") or {}
    merged_parameters = {**carried["parameters"], **published_parameters}
    # `extra_body` is a nested dict: a shallow merge would let a published
    # `extra_body` replace the carried one wholesale and silently drop
    # `chat_template_kwargs.enable_thinking`. Published wins per key.
    published_extra_body = published_parameters.get("extra_body")
    if published_extra_body is None:
        # An explicit `extra_body: null` means "nothing published", not "clear
        # the carried pin": the flat merge above would otherwise store None.
        merged_parameters["extra_body"] = carried["parameters"]["extra_body"]
    elif isinstance(published_extra_body, dict):
        try:
            merged_parameters["extra_body"] = ModelOption.merge_extra_body(
                carried["parameters"]["extra_body"], published_extra_body
            )
        except TypeError as exc:
            reason = f"published parameters.extra_body is malformed: {exc}"
            logger.error(
                "Adapter function '%s' (decision capability) not registered: "
                "%s. Other adapters from this source are unaffected.",
                name,
                reason,
            )
            return None, reason
    else:
        reason = (
            "published parameters.extra_body must be a mapping, got "
            f"{type(published_extra_body).__name__}"
        )
        logger.error(
            "Adapter function '%s' (decision capability) not registered: %s. "
            "Other adapters from this source are unaffected.",
            name,
            reason,
        )
        return None, reason
    merged["parameters"] = merged_parameters
    if carried_fields_used:
        merged["provenance"] = (
            f"{_CARRIED_DECIDER_PROVENANCE}; carried fields applied: "
            f"{sorted(carried_fields_used)}"
        )

    # A one-token readout must never be constrained by guided decoding.
    transformations = merged.get("transformations")
    if isinstance(transformations, list) and any(
        isinstance(t, dict) and t.get("type") == "first_token" for t in transformations
    ):
        merged["response_format"] = None

    invalid_reason = _validate_merged_decider_config(merged)
    if invalid_reason is not None:
        logger.error(
            "Adapter function '%s' (decision capability) not registered: %s. "
            "Other adapters from this source are unaffected.",
            name,
            invalid_reason,
        )
        return None, invalid_reason
    return merged, None


def _io_config_root(model_path: pathlib.Path) -> pathlib.Path:
    """Directory an `io_config` file must resolve inside.

    Normally the model directory itself, so a symlink cannot point outside it.
    A Hugging Face cache snapshot (`<repo>/snapshots/<revision>/`) is the
    exception: every file in it is a symlink into the sibling `<repo>/blobs/`
    directory, so the allowed root is the repo's cache directory. Only that
    exact layout is widened: the parent must be named `snapshots` and a
    `blobs` directory must sit beside it. This matches the root the hub-id
    download path already allows.

    Args:
        model_path: The model directory passed by the caller.

    Returns:
        The resolved directory that `io_config` files must stay within.
    """
    resolved = model_path.resolve()
    repo_cache = resolved.parent.parent
    if resolved.parent.name == "snapshots" and (repo_cache / "blobs").is_dir():
        return repo_cache
    return resolved


def _discover_embedded_adapters(
    source: str,
    *,
    revision: str = "main",
    cache_dir: str | None = None,
    intrinsic_name: str | None = None,
) -> tuple[list[tuple["_AdapterCore", dict]], dict[str, str]]:
    """Discover embedded adapter functions from a Granite Switch source.

    Non-shim equivalent of `EmbeddedIntrinsicAdapter.from_source()`: returns
    composed `Adapter` instances instead of the deprecated shim (Epic #929,
    issue #1144). Reuses the shim's discovery, `adapter_index.json`/`io.yaml`
    parsing, and hub-snapshot materialization internally; including its
    path-escape guard; rather than duplicating that logic, and lifts out the
    already-correct `identity`/`io_contract`/`weights` triple each shim
    instance built.

    The raw parsed `io.yaml` config is returned alongside each adapter because
    a composed `Adapter` has no field for it (that's shim-only state) and it
    cannot be cheaply re-derived later; callers that need it at generation
    time (see `LocalHFBackend`/`OpenAIBackend`'s `_generate_from_intrinsic`)
    must cache it themselves, keyed by `_composed_adapter_key`.

    Args:
        source (str): Local path to a model directory, or a Hugging Face Hub
            repo ID (e.g. `"ibm-granite/granite-switch-micro"`).
        revision (str): Git revision (only used for Hub downloads).
        cache_dir (str | None): Cache directory (only used for Hub downloads).
        intrinsic_name (str | None): If provided, only load the adapter
            matching this adapter function name. `None` loads all adapters.

    Returns:
        tuple[list[tuple[_AdapterCore, dict]], dict[str, str]]: One
            `(adapter, io_yaml_config)` pair per registrable entry in the
            index, plus a dict mapping skipped adapter names to their skip
            reasons (currently only an invalid merged decider config).
    """
    if not pathlib.Path(source).is_dir():
        MelleaLogger.get_logger().info(
            "Discovering embedded adapter functions from %r at revision %r.",
            source,
            revision,
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        shims = EmbeddedIntrinsicAdapter.from_source(
            source,
            revision=revision,
            cache_dir=cache_dir,
            intrinsic_name=intrinsic_name,
        )
    results: list[tuple[_AdapterCore, dict]] = []
    skip_reasons: dict[str, str] = {}
    merged_invocation_text: str | None = None
    for shim in shims:
        config, invalid_reason = _apply_carried_decider_config(
            shim.identity.name, shim.config
        )
        if config is None:
            assert invalid_reason is not None
            skip_reasons[shim.identity.name] = invalid_reason
            continue
        if shim.identity.name == DECISION_CAPABILITY and isinstance(
            config.get("invocation_text"), str
        ):
            merged_invocation_text = config["invocation_text"]
        results.append(
            (
                _AdapterCore(
                    identity=shim.identity,
                    io_contract=shim.io_contract,
                    weights=shim.weights,
                ),
                config,
            )
        )
    if merged_invocation_text is not None:
        _check_decision_template_invocation(
            source,
            revision=revision,
            cache_dir=cache_dir,
            merged_invocation_text=merged_invocation_text,
        )
    return results, skip_reasons


def get_adapter_for_intrinsic(
    intrinsic_name: str,
    intrinsic_adapter_types: list[AdapterType] | tuple[AdapterType, ...],
    available_adapters: dict[str, T],
) -> T | None:
    """Find an adapter from a dict of available adapters based on the adapter function name and its allowed adapter types.

    Args:
        intrinsic_name (str): The name of the adapter function, e.g. `"answerability"`.
        intrinsic_adapter_types (list[AdapterType] | tuple[AdapterType, ...]): The
            adapter types allowed for this adapter function, e.g.
            `[AdapterType.ALORA, AdapterType.LORA]`.
        available_adapters (dict[str, T]): The available adapters to choose from;
            maps `adapter.qualified_name` to the adapter object.

    Returns:
        T | None: The first matching adapter found, or `None` if no match exists.
    """
    adapter = None
    for adapter_type in intrinsic_adapter_types:
        qualified_name = f"{intrinsic_name}_{adapter_type.value}"
        adapter = available_adapters.get(qualified_name)
        if adapter is not None:
            break

    return adapter


def _alora_invocation_sequence_present(
    prompt_token_ids: Sequence[int], invocation_tokens: Sequence[int]
) -> bool:
    """Return whether an aLoRA's declared invocation token sequence occurs in a prompt.

    An aLoRA only activates from the point where PEFT finds this exact
    contiguous token-id subsequence in the assembled prompt
    (`peft.tuners.lora.variants.calculate_alora_offsets`). When it is absent,
    PEFT sets the activation offset to `None` and generation silently runs
    against the base model with no error and no warning. This
    function is the check Mellea runs itself, since PEFT does not surface the
    mismatch: it holds both the loaded adapter's declared sequence and the
    prompt it just assembled.

    The check is deliberately a token-id subsequence search over the fully
    assembled prompt, not a substring search over instruction text: tokeniser
    merges across a text boundary (e.g. `>` and the following `:` fusing into
    one token) can make a sequence that looks present in the source text
    absent from the actual token ids, and vice versa for sequences supplied by
    the chat template rather than the instruction.

    Matches anywhere the sequence occurs, not only at the offset PEFT would
    actually activate from (PEFT itself uses the *last* match; see
    `calculate_alora_offsets`). If the exact token sequence happens to occur
    earlier in the prompt (e.g. literally present in older conversation
    turns), this returns `True` even though the adapter would activate from
    that earlier, unintended position rather than not at all. That is a
    different failure mode (wrong span, not no activation) and is not
    detected here.

    Args:
        prompt_token_ids: Token ids of the assembled prompt that will be sent
            to the model.
        invocation_tokens: The aLoRA's declared `alora_invocation_tokens`.

    Returns:
        bool: `True` if `invocation_tokens` is empty (nothing declared to look
        for) or occurs contiguously in `prompt_token_ids`; `False` otherwise.
    """
    inv = list(invocation_tokens)
    if not inv:
        return True
    prompt = list(prompt_token_ids)
    n = len(inv)
    if n > len(prompt):
        return False
    return any(prompt[i : i + n] == inv for i in range(len(prompt) - n + 1))


def _fire_phase_complete_hook(name: str, phase: str, duration_ms: float) -> None:
    """Fire the `adapter_function_phase_complete` metric hook for a phase that already ran.

    Split out of `_run_adapter_phase` so a caller that must guarantee cleanup
    after a phase's side effect; e.g. `adapter_scope` guaranteeing
    `deactivate()` runs once `activate()` has succeeded; can run the side
    effect and this hook fire under separate exception handling. A hook-dispatch
    failure is logged and ignored: observability must not turn a completed
    lifecycle phase into an operation failure.

    Args:
        name: Adapter function name, used as the metric's `name` field.
        phase: Lifecycle phase name; must be a valid
            `AdapterFunctionPhaseCompletePayload.phase` value.
        duration_ms: Wall-clock duration of the phase, in milliseconds.
    """
    if not has_plugins(HookType.ADAPTER_FUNCTION_PHASE_COMPLETE):
        return
    from ...plugins.hooks.adapter_function import AdapterFunctionPhaseCompletePayload

    payload = AdapterFunctionPhaseCompletePayload(
        name=name, phase=phase, duration_ms=duration_ms
    )
    try:
        hook_coro = invoke_hook(HookType.ADAPTER_FUNCTION_PHASE_COMPLETE, payload)
        _run_async_in_thread(hook_coro)
    except Exception:
        MelleaLogger.get_logger().warning(
            f"adapter_function_phase_complete hook dispatch failed for {name!r} "
            f"during {phase!r}; ignoring so it does not turn a completed phase "
            "into an operation failure.",
            exc_info=True,
        )


def _run_adapter_phase(name: str, phase: str, phase_fn: Callable[[], None]) -> None:
    """Run one lifecycle phase and fire its phase-complete metric hook.

    Fires the hook only; it does not open a span. Span production belongs to a
    plugin (#1464, #1466), not to code under `mellea/backends/`.

    The hook fires **only when the phase succeeds**, matching the name of
    `ADAPTER_FUNCTION_PHASE_COMPLETE`: a phase that raised did not complete. If
    `phase_fn` raises, the exception propagates and no phase event is emitted, so
    a consumer reconciling phase counts against invocation counts will see the
    failure only at invocation level, where `outcome` and `error` carry it.

    Args:
        name: Adapter function name, used as the metric's `name` field.
        phase: Lifecycle phase name; must be a valid
            `AdapterFunctionPhaseCompletePayload.phase` value.
        phase_fn: The zero-argument callable implementing the phase (e.g.
            `adapter.weights.activate`).
    """
    started_at = time.monotonic()
    phase_fn()
    _fire_phase_complete_hook(name, phase, (time.monotonic() - started_at) * 1000.0)


def _fire_invocation_complete(
    *,
    name: str,
    revision: str | None,
    binding_type: str,
    adapter_type: str,
    outcome: Literal["success", "schema_error", "error"],
    error: BaseException | None,
) -> None:
    """Fire the `adapter_function_invocation_complete` metric hook.

    Args:
        name: Adapter function name.
        revision: Catalog revision of the adapter, or `None` if unpinned.
        binding_type: Weight-binding reality the adapter ran under.
        adapter_type: Adapter mechanism (e.g. `"lora"`, `"alora"`).
        outcome: Invocation outcome.
        error: The exception raised during invocation, or `None` on success.
    """
    if not has_plugins(HookType.ADAPTER_FUNCTION_INVOCATION_COMPLETE):
        return
    from ...plugins.hooks.adapter_function import (
        AdapterFunctionInvocationCompletePayload,
    )

    payload = AdapterFunctionInvocationCompletePayload(
        name=name,
        revision=revision,
        binding_type=binding_type,
        adapter_type=adapter_type,
        outcome=outcome,
        error=error,
    )
    hook_coro = invoke_hook(HookType.ADAPTER_FUNCTION_INVOCATION_COMPLETE, payload)
    _run_async_in_thread(hook_coro)


# The full adapter-input surface `add_adapter` advertises. The legacy abc
# `Adapter` (LocalFile/PEFT) and the core dataclass adapter (`_AdapterCore`,
# Embedded/ServerMediated) are disjoint hierarchies, so the accepted type is
# their union. Concrete backends accept this union and raise `TypeError` for the
# adapter realities they do not implement; the same "reject unsupported reality"
# contract the reality-specific verbs use. See the module note on the mixin-vs-
# generic trade-off for why this is a runtime, not a type-parameter, guarantee.
AdapterInput: TypeAlias = Adapter | _AdapterCore | LocalFileBinding


class AdapterMixin(Backend, abc.ABC):
    """Mixin class for backends capable of utilizing adapters.

    Three verbs are universal across every adapter reality (LocalFile/PEFT,
    Embedded/Granite Switch, ServerMediated): `base_model_name`,
    `add_adapter`, and `list_adapters`. The remaining five verbs are
    reality-specific; a concrete backend overrides only the verb(s) matching
    its own reality; the others keep raising `NotImplementedError`.

    Attributes:
        base_model_name (str): The short model name used to identify adapter
            variants (e.g. `"granite-3.3-8b-instruct"` for
            `"ibm-granite/granite-3.3-8b-instruct"`).
    """

    _supports_composed_adapters: ClassVar[bool] = False
    """Whether this backend's `add_adapter` understands a composed `Adapter`.

    Opt-in, defaulting to `False`: `_uses_embedded_adapters` predates the
    composed `Adapter` contract (Epic #929, issue #1144), so a third-party
    subclass can already be reachable from `resolve_adapter()`'s embedded
    branch while its `add_adapter` only knows the deprecated
    `EmbeddedIntrinsicAdapter` shim's attribute shape. Signature inspection
    cannot distinguish "modern" from "legacy" here; a `**kwargs` catch-all
    present for unrelated reasons, or a pre-existing `config` parameter
    meaning something else entirely, both read as "accepts config=". Only an
    explicit declaration can carry that fact; `LocalHFBackend` and
    `OpenAIBackend` both set this to `True`.
    """

    _adapter_skip_reasons: dict[str, str] = {}
    """Reasons a capability was skipped at embedded-adapter discovery time.

    Class-level empty-dict default: writers assign a fresh instance dict
    (never mutate in place), so the class default stays read-only.
    Populated by the discovery paths (`resolve_adapter` itself and
    `register_embedded_adapter_model`) when a merged decider config fails
    validation or a template check fails.
    """

    # ---- Universal verbs (every adapter reality) ----

    @property
    @abc.abstractmethod
    def base_model_name(self) -> str:
        """Return the short model name used for adapter variant lookup.

        Returns:
            str: The base model name (e.g. `"granite-3.3-8b-instruct"`).
        """

    @abc.abstractmethod
    def add_adapter(self, adapter: AdapterInput, *, config: dict | None = None) -> None:
        """Register an adapter with this backend so it can be loaded later.

        The adapter must not already have been added to a different backend.
        Concrete backends accept the full `AdapterInput` union but raise
        `TypeError` for adapter realities they do not implement (e.g. a PEFT
        backend rejects an embedded adapter), so a statically valid call may
        still be rejected at runtime.

        `config` is the raw io.yaml mapping for a composed `Adapter`/
        `_AdapterCore` whose `weights` is an `EmbeddedBinding` or
        `ServerMediatedBinding`. Those realities do not retain the raw
        configuration required by the legacy rewriter, so it must be supplied
        at registration rather than being fetched lazily like a
        `LocalFileBinding`'s io.yaml.

        Args:
            adapter (AdapterInput): The adapter to register with this backend.
            config (dict | None): Raw io.yaml config for a composed
                `EmbeddedBinding` or `ServerMediatedBinding` adapter. Ignored
                (and rejected) for every other adapter reality.

        Raises:
            TypeError: If `adapter` belongs to a reality this backend does not
                support, or `config` is given for a reality other than a
                composed `EmbeddedBinding` or `ServerMediatedBinding` adapter.
            ValueError: If `adapter.weights` is an `EmbeddedBinding` or
                `ServerMediatedBinding` and `config` is not given —
                registering it without a config would make it discoverable
                but permanently unable to generate.
        """

    @abc.abstractmethod
    def list_adapters(self) -> list[str]:
        """Return the qualified names of all adapters registered with this backend.

        Returns:
            list[str]: Qualified adapter names for all adapters that have been
                registered via `add_adapter`.
        """
        ...

    # ---- Reality-specific verbs ----

    def load_peft_adapter(self, adapter_qualified_name: str) -> None:
        """Load a previously registered PEFT adapter into the underlying model.

        LocalFile/PEFT reality only (e.g. a locally hosted Hugging Face
        model). The adapter must have been registered via `add_adapter`
        before calling this method.

        Args:
            adapter_qualified_name (str): The `adapter.qualified_name` of the
                adapter to load.

        Raises:
            NotImplementedError: If this backend's adapter reality is not
                LocalFile/PEFT.
        """
        raise NotImplementedError(
            f"Backend type {type(self)} does not support load_peft_adapter()."
        )

    def unload_peft_adapter(self, adapter_qualified_name: str) -> None:
        """Unload a previously loaded PEFT adapter from the underlying model.

        LocalFile/PEFT reality only (e.g. a locally hosted Hugging Face
        model).

        Args:
            adapter_qualified_name (str): The `adapter.qualified_name` of the
                adapter to unload.

        Raises:
            NotImplementedError: If this backend's adapter reality is not
                LocalFile/PEFT.
        """
        raise NotImplementedError(
            f"Backend type {type(self)} does not support unload_peft_adapter()."
        )

    def remove_adapter(self, adapter_qualified_name: str) -> None:
        """Deregister a previously added adapter, freeing its qualified name for reuse.

        The inverse of `add_adapter()`. LocalFile/PEFT bindings call this from
        `LocalFileBinding.release()` after unloading their weights. Embedded
        backends use it to clear their registration and cached configuration;
        their weights are already part of the served model, so there is no
        unload step.

        Args:
            adapter_qualified_name (str): The `adapter.qualified_name` of the
                adapter to deregister.

        Raises:
            NotImplementedError: If this backend's adapter reality does not
                support deregistration.
        """
        raise NotImplementedError(
            f"Backend type {type(self)} does not support remove_adapter()."
        )

    def activate_peft_adapter(self, adapter_qualified_name: str) -> None:
        """Switch a previously loaded PEFT adapter on for subsequent generation.

        LocalFile/PEFT reality only (e.g. a locally hosted Hugging Face
        model). The adapter must have been loaded via `load_peft_adapter`
        before calling this method.

        Args:
            adapter_qualified_name (str): The `adapter.qualified_name` of the
                adapter to activate.

        Raises:
            NotImplementedError: If this backend's adapter reality is not
                LocalFile/PEFT.
        """
        raise NotImplementedError(
            f"Backend type {type(self)} does not support activate_peft_adapter()."
        )

    def deactivate_peft_adapter(self, adapter_qualified_name: str) -> None:
        """Switch off any active PEFT adapter so generation uses the base model.

        LocalFile/PEFT reality only (e.g. a locally hosted Hugging Face
        model).

        Args:
            adapter_qualified_name (str): The `adapter.qualified_name` of the
                adapter to deactivate. Accepted for symmetry with
                `activate_peft_adapter`; the underlying primitive clears all
                active PEFT adapters regardless of name.

        Raises:
            NotImplementedError: If this backend's adapter reality is not
                LocalFile/PEFT.
        """
        raise NotImplementedError(
            f"Backend type {type(self)} does not support deactivate_peft_adapter()."
        )

    def _adapter_activation_lock(
        self,
    ) -> contextlib.AbstractContextManager[bool | None]:
        """Exclusivity lock to hold while calling activate/deactivate and dict writes.

        Default is a no-op (`contextlib.nullcontext()`). Backends whose
        activation verbs mutate shared, non-thread-safe state (e.g.
        `LocalHFBackend`'s underlying PEFT model) override this to return
        their own lock, so callers like `LocalFileBinding.activate()` get
        the same exclusivity `_generate_with_adapter_lock` relies on.
        `add_adapter()`'s registration-dict writes also hold it, briefly.

        Innermost in the adapter lock order
        (`_adapter_resolve_lock` -> `binding._lifecycle_lock` ->
        `_adapter_activation_lock`). Never hold this lock across a
        `WeightsBinding` lifecycle verb (`prepare`/`activate`/`deactivate`/
        `release`) or across any I/O; those take `_lifecycle_lock`, and
        holding activation across them inverts the order and can deadlock
        against a concurrent lifecycle call on the same binding.

        A code path already holding this lock can re-enter it: on
        `LocalHFBackend`, `_generate_intrinsic_with_adapter_scope` holds
        `_generation_lock` for the whole generation, and the
        `_IntrinsicPeftBinding` verbs it drives through `adapter_scope()`
        take `_adapter_activation_lock()` again on the same thread. An
        override must therefore return a reentrant lock (`threading.RLock`,
        as `LocalHFBackend` does); a plain `threading.Lock` here is a real
        deadlock hazard.
        """
        return contextlib.nullcontext()

    def _adapter_resolve_lock(self) -> contextlib.AbstractContextManager[bool | None]:
        """Exclusivity lock for discovery-plus-registration orchestration.

        Default is a no-op (`contextlib.nullcontext()`). Held by
        `resolve_adapter()` and `register_embedded_adapter_model()` across
        their catalogue/Hugging Face Hub discovery I/O and their
        `add_adapter()` calls.

        Outermost in the adapter lock order
        (`_adapter_resolve_lock` -> `binding._lifecycle_lock` ->
        `_adapter_activation_lock`). Deliberately a separate lock from
        `_adapter_activation_lock`, not a reuse of it: the activation lock is
        taken *inside* every `WeightsBinding` lifecycle verb, so holding it
        across `add_adapter()`; which can call those verbs; would invert
        the order and deadlock against a concurrent `prepare()`/`release()`
        on the same binding.

        Must not be acquired while already holding
        `_adapter_activation_lock()`; that reverses the order. No
        production caller does: `resolve_adapter()` runs before the backend
        takes its generation lock, not after (see
        `mellea/stdlib/components/intrinsic/_util.py`).

        An override must return a reentrant lock (`threading.RLock`) for the
        same same-thread-reentry reason `_adapter_activation_lock()`
        documents.

        Known limitation: a `HookType.ADAPTER_FUNCTION_PHASE_COMPLETE`
        subscriber must not synchronously call `resolve_adapter()`,
        `register_embedded_adapter_model()`, or `add_adapter()` with a
        composed `Adapter` on this same backend. `LocalFileBinding.prepare()`
        fires that hook via `_run_async_in_thread`, which blocks the calling
        thread on a dedicated event-loop thread's result; if this lock is
        held at that point (as it is for the whole body of the three methods
        above) and the hook's handler re-enters one of them on the same
        backend, the event-loop thread blocks on this lock while the original
        thread blocks waiting for the event-loop thread: deadlock. No shipped
        caller does this today.
        """
        return contextlib.nullcontext()

    def _add_embedded_adapter_compat(
        self, adapter: "_AdapterCore", config: dict
    ) -> None:
        """Register a composed embedded Adapter, tolerating a pre-#1144 `add_adapter`.

        `AdapterMixin.add_adapter` gained a `config` keyword-only parameter in
        Epic #929, issue #1144 (needed because a composed `Adapter` has no
        field to carry a shim's `.config`). `_uses_embedded_adapters` predates
        that: a third-party `AdapterMixin` subclass written before this
        parameter existed, but already supporting the Embedded reality, is
        reachable here and would raise
        `TypeError: add_adapter() got an unexpected keyword argument 'config'`
        ; or worse, silently misbehave, if it happens to already accept a
        `config=`/`**kwargs` for unrelated reasons and gets handed a composed
        `Adapter` its body doesn't know how to read (it has no
        `.qualified_name`/`.config`/`.technology`, unlike the shim). Gated on
        `_supports_composed_adapters`, not signature inspection, for exactly
        that reason: a parameter name can't tell you whether the method body
        understands the new object shape.

        Args:
            adapter: The composed `Adapter` to register.
            config: Raw io.yaml config for `adapter`.
        """
        if self._supports_composed_adapters:
            self.add_adapter(adapter, config=config)
            return
        # Legacy subclass predating the composed-Adapter contract: it only
        # knows the deprecated EmbeddedIntrinsicAdapter shim, which carries
        # its own .config and needs no config= parameter.
        self.add_adapter(
            EmbeddedIntrinsicAdapter(
                adapter.identity.name, config, technology=adapter.identity.adapter_type
            )
        )

    def resolve_adapter(self, name: str) -> _AdapterCore:
        """Find or lazily register an adapter by capability name.

        Default implementation preserves Phase 0 behaviour, using the internal
        `_added_adapters` dict that concrete backends maintain.  Override in
        Phase 2 (see epic #929) to implement proper lifecycle management.

        Args:
            name (str): Capability name (e.g. `"answerability"`).

        Returns:
            _AdapterCore: The registered adapter with the given capability.

        Raises:
            ValueError: If the backend has no model ID.
            KeyError: If the adapter cannot be found after registration.

        Note:
            A cold first resolve holds `_adapter_resolve_lock()` across any
            Hugging Face Hub download it triggers, so it can stall every
            other caller of *that* lock (other resolves,
            `register_embedded_adapter_model()`); including a resolve on
            the asyncio event-loop thread inside a gathered set of
            coroutines. It does **not** stall generation or activation:
            `_adapter_resolve_lock` is a separate lock from
            `_adapter_activation_lock`, deliberately (see the lock-order note
            on `_adapter_activation_lock()`). A warm cache after the first
            resolve avoids the download either way.
        """
        found = self._find_adapter(name)
        if found is not None:
            return found

        base = self.base_model_name
        if base is None:
            raise ValueError(
                f"Backend has no model ID; cannot resolve adapter {name!r}"
            )

        # add_adapter()'s own duplicate check is an unguarded read-then-write on
        # _added_adapters, and catch_warnings() below mutates thread-unsafe global
        # filter state; both race under concurrent first-time resolves for the
        # same name. `_adapter_resolve_lock()` closes both: a no-op by default,
        # and each concrete backend's reentrant lock otherwise. Deliberately not
        # `_adapter_activation_lock()`: that lock is taken *inside*
        # `add_adapter()`'s composed-LocalFileBinding branch around the dict
        # writes only, and inside every `WeightsBinding` lifecycle verb; holding
        # it here, across the whole discover-and-register loop below (which calls
        # `add_adapter()`, which can call `binding.prepare()`), would invert the
        # adapter lock order and deadlock against a concurrent
        # `prepare()`/`release()` on the same binding.
        # Suppress DeprecationWarning: the shim constructors warn user-facing code,
        # not internal registration paths.
        with self._adapter_resolve_lock(), warnings.catch_warnings():
            # Re-check now the lock is held: a concurrent resolve may have already
            # registered this name. Without this, the loser redundantly re-fetches
            # and then hits the backend's own duplicate guard, which logs a
            # misleading "client code ... not idempotent" warning.
            found = self._find_adapter(name)
            if found is not None:
                return found

            warnings.simplefilter("ignore", DeprecationWarning)
            if getattr(self, "_uses_embedded_adapters", False):
                repo_id = (
                    getattr(self, "_adapter_source", None)
                    or getattr(self, "_model_id", None)
                    or base
                )
                # Composed Adapter, not the deprecated EmbeddedIntrinsicAdapter
                # shim (Epic #929, issue #1144). Valid only for backends whose
                # add_adapter supports the Embedded/Granite Switch reality
                # (currently OpenAIBackend and LocalHFBackend when configured
                # with load_embedded_adapters=True).
                #
                # Passing config= lets add_adapter() cache it atomically with
                # registration, gated behind its own duplicate-key guard; a
                # refused duplicate (a different object already holds the
                # key) therefore never reaches the config write, so this
                # can't clobber a live adapter's cached config the way a
                # register-then-separately-cache sequence could. Routed
                # through _add_embedded_adapter_compat, not called directly,
                # for a third-party AdapterMixin subclass whose add_adapter
                # predates config= (see that method's docstring).
                discovered, skip_reasons = _discover_embedded_adapters(
                    repo_id,
                    revision=getattr(self, "_adapter_revision", "main"),
                    intrinsic_name=name,
                )
                if skip_reasons:
                    self._adapter_skip_reasons = {
                        **self._adapter_skip_reasons,
                        **skip_reasons,
                    }
                for a, config in discovered:
                    self._add_embedded_adapter_compat(a, config)
            else:
                # AdapterType.LORA is the pre-Phase-1 default (mirrors old _util.py).
                # Every current catalog entry supports LORA.  Phase 2 (see epic #929)
                # will select the type from catalog availability instead of hardcoding.
                metadata = fetch_intrinsic_metadata(name)
                if self._supports_composed_adapters:
                    # Composed Adapter, not the deprecated IntrinsicAdapter shim
                    # (Epic #929, issue #1144).
                    self.add_adapter(
                        _AdapterCore(
                            identity=Identity(
                                name=name,
                                adapter_type="lora",
                                capability=metadata.effective_capability,
                            ),
                            io_contract=get_io_contract(name),
                            weights=LocalFileBinding(
                                name=name,
                                adapter_type=AdapterType.LORA,
                                repo_id=metadata.repo_id,
                                revision=metadata.revision,
                            ),
                        )
                    )
                else:
                    # Legacy subclass predating the composed-Adapter contract
                    # (same reasoning as _add_embedded_adapter_compat): its
                    # add_adapter only knows the deprecated IntrinsicAdapter
                    # shim's attribute shape, not a composed Adapter's. This
                    # is exactly what this branch passed before Epic #929,
                    # issue #1144; IntrinsicAdapter is dual-shaped (both
                    # LocalHFAdapter, for `.qualified_name`/`.backend`/`.path`,
                    # and _AdapterCore, so `_find_adapter` below and this
                    # method's own `-> _AdapterCore` return type still hold).
                    self.add_adapter(
                        IntrinsicAdapter(
                            name, adapter_type=AdapterType.LORA, base_model_name=base
                        )
                    )

        found = self._find_adapter(name)
        if found is not None:
            return found

        # `_find_adapter` only matches `_AdapterCore` entries. If registration
        # above silently failed because a `LocalFileBinding` already claims a
        # colliding qualified name (they share `f"{name}_{type}"` with
        # `IntrinsicAdapter`), say so; the alternative is an opaque KeyError
        # that gives no hint the two registration paths collided.
        # list(...): same concurrent-mutation hazard as `_find_adapter`; snapshot
        # before iterating rather than holding a live view over `_added_adapters`.
        added = list(getattr(self, "_added_adapters", {}).items())
        blocking = next(
            (
                v
                for k, v in added
                if k.startswith(f"{name}_") and not isinstance(v, _AdapterCore)
            ),
            None,
        )
        if blocking is not None:
            blocking_name = getattr(blocking, "qualified_name", None)
            raise KeyError(
                f"Adapter {name!r} not found after registration: a "
                f"{type(blocking).__name__} is already registered under "
                f"{blocking_name!r}, which collides with {name!r}'s auto-registration "
                "path. LocalFileBinding and resolve_adapter()/intrinsic-helper "
                "registrations share the same qualified-name key space on this "
                "backend and cannot both claim it."
            )

        skip_reason = self._adapter_skip_reasons.get(name)
        if skip_reason:
            raise KeyError(
                f"Adapter {name!r} not found after registration: {skip_reason}"
            )
        raise KeyError(f"Adapter {name!r} not found after registration")

    @contextlib.contextmanager
    def adapter_scope(self, adapter: "_AdapterCore | None"):  # type: ignore[type-arg]
        """Context manager wrapping adapter activation and deactivation.

        A no-op when `adapter` is `None`. Otherwise: activates
        `adapter.weights`, yields, then always deactivates; even if the `with`
        body raises. Each phase fires `ADAPTER_FUNCTION_PHASE_COMPLETE`, and
        `ADAPTER_FUNCTION_INVOCATION_COMPLETE` fires on the way out, carrying the
        overall outcome.

        This method fires hooks only; it does not open spans. Span production is a
        plugin's job (see #1464 for the rule and #1466 for the adapter-function
        spans), and the `ADAPTER_FUNCTION_*` family currently has no start hook for
        a plugin to open a span on. Hook dispatch goes through
        `_run_async_in_thread` (no timeout): the dispatching call blocks the
        calling thread, but the hook coroutine itself runs on the shared
        `_EventLoopHandler` event-loop thread. A subscriber that blocks on
        something the dispatching thread is holding deadlocks rather than
        merely stalls; e.g. on `LocalHFBackend`, an intrinsic caller holds
        `_generation_lock` across the whole scope, so a subscriber that
        re-enters any `_generation_lock` path blocks the event-loop thread
        while its owner waits on that same event loop, and reentrance cannot
        bridge the gap. Even without such re-entry, a slow or blocking-mode
        `ADAPTER_FUNCTION_*` subscriber delays whatever holds this scope open.

        `deactivate()` is guarded on `activate()`'s own side effect having
        completed, not on the activate phase's hook dispatch also succeeding.
        If a plugin subscribed to `ADAPTER_FUNCTION_PHASE_COMPLETE` raises after
        `activate()` already flipped the adapter on, `deactivate()` still runs —
        telemetry must not be able to strand the adapter active.

        Not atomic across the whole scope **by itself**: `_adapter_activation_lock()`
        is held only inside each of `activate()`/`deactivate()`'s own verb calls
        (see `LocalFileBinding.activate`), not for the `with` body in between.
        Two concurrent `adapter_scope()` calls on one backend can therefore
        interleave; one thread's body can run while a different adapter is
        active, activated by another thread's call; unless the caller closes
        that gap itself. Widening *this method's own* lock to span the whole
        scope was tried and reverted: it deadlocks the moment the body does
        real async generation from the thread that opened the scope, because
        that work runs on the shared event-loop thread while this thread holds
        the lock; a same-thread `RLock` doesn't help across threads.

        `LocalHFBackend._generate_intrinsic_with_adapter_scope` is the reference
        example of a caller that *does* close the gap for its own call site: it
        holds `_generation_lock` around the entire scope, which is safe there
        only because the scope *body* is fully synchronous end to end and
        does no async generation work on the event loop (its only loop
        traffic during the scope is the hook dispatches described above —
        one-way submissions, not re-entry into this backend); concurrent
        invocations simply land on different threads and serialise on the
        lock, rather than one thread holding it while another does async work
        on the loop. A caller whose
        body awaits work that re-enters generation on another thread must not
        widen a lock this way; that reproduces the deadlock above.

        A caller composing `adapter_scope()` with `LocalHFBackend`'s *standard*
        (non-intrinsic) generation path still silently ignores it: that path
        (`_generate_with_adapter_lock`) always deactivates any adapter before
        generating, so wrapping `generate_from_context()` in `adapter_scope()`
        activates the adapter, generates against the base model anyway, then
        deactivates. Pre-existing, not specific to the intrinsic path this
        method now supports.

        `AdapterFunctionMetricsPlugin` in
        `mellea/telemetry/metrics_plugins.py` emits the adapter-function
        metrics; their instruments and attributes are defined in
        `mellea/telemetry/metrics.py`.

        Args:
            adapter: The adapter to activate, or `None` (no-op).

        Raises:
            TypeError: `adapter.weights` is not a `WeightsBinding` (e.g. an
                `EmbeddedBinding`, which has no activate()/deactivate() to
                scope; call its `apply_activation()` directly instead).
            BaseException: An error raised by activation, the `with` body, or
                deactivation. If both the body and deactivation fail, the body
                error remains primary and the deactivation error is chained.
        """
        if adapter is None:
            yield
            return

        name = adapter.identity.name
        # Prefer `resolved_revision()` over the raw `.revision` attribute: a
        # lazily-resolved binding (`revision=None`) still downloads and runs
        # against the catalogue's pinned SHA, so reporting the unresolved
        # `None` would mislabel an effectively-pinned invocation as unpinned.
        # `resolved_revision()` only exists on `LocalFileBinding`, not the
        # `WeightsBinding` base, so both the lookup and the call are guarded.
        revision: str | None
        if isinstance(adapter.weights, LocalFileBinding):
            try:
                revision = adapter.weights.resolved_revision()
            except Exception:
                revision = adapter.weights.revision
        else:
            revision = cast(str | None, getattr(adapter.weights, "revision", None))
        binding_type = adapter.weights.binding_type
        adapter_type = adapter.identity.adapter_type

        # adapter_scope drives the WeightsBinding lifecycle (activate/deactivate);
        # a binding with no lifecycle (e.g. EmbeddedBinding) activates through its
        # own apply_activation() instead (issue #1142) and never reaches this scope.
        if not isinstance(adapter.weights, WeightsBinding):
            raise TypeError(
                f"adapter_scope() requires a WeightsBinding-backed adapter; "
                f"{binding_type!r} bindings have no activate()/deactivate() to "
                "scope. Call apply_activation() directly instead."
            )

        outcome: Literal["success", "schema_error", "error"] = "success"
        exception: BaseException | None = None
        activated = False
        body_exception: BaseException | None = None
        try:
            started_at = time.monotonic()
            try:
                adapter.weights.activate()
                activated = True
                _fire_phase_complete_hook(
                    name, "activate", (time.monotonic() - started_at) * 1000.0
                )
                try:
                    yield
                except BaseException as exc:
                    body_exception = exc
                    raise
            finally:
                if activated:
                    try:
                        _run_adapter_phase(
                            name, "deactivate", adapter.weights.deactivate
                        )
                    except BaseException as deactivate_exc:
                        if body_exception is None:
                            raise
                        body_exception.add_note(
                            "Adapter deactivation also failed: "
                            f"{type(deactivate_exc).__name__}: {deactivate_exc}"
                        )
        except AdapterSchemaMismatchError as exc:
            # Distinct from a generic error: this is the schema-drift signal the
            # `parse_failures` counter exists to detect, so collapsing it into
            # "error" would leave that counter permanently at zero. Reachable
            # today; `adapter_scope` is public, so a caller can parse inside the
            # scope; and it becomes the common case once #1465 moves generation
            # and parsing in here.
            outcome = "schema_error"
            exception = exc
            raise
        except BaseException as exc:
            outcome = "error"
            exception = exc
            raise
        finally:
            # A hook-dispatch failure here must not replace or mask the real
            # outcome computed above; that would turn a clean `with` block
            # into a thrown error, or swap a genuine body exception for a
            # telemetry-plumbing one. Log and swallow instead.
            try:
                _fire_invocation_complete(
                    name=name,
                    revision=revision,
                    binding_type=binding_type,
                    adapter_type=adapter_type,
                    outcome=outcome,
                    error=exception,
                )
            except Exception:
                MelleaLogger.get_logger().warning(
                    f"adapter_function_invocation_complete hook dispatch failed for "
                    f"{name!r}; ignoring so it doesn't mask the real outcome "
                    f"({outcome!r}).",
                    exc_info=True,
                )

    def _find_adapter(
        self, capability: str, adapter_types: tuple[str, ...] | None = None
    ) -> "_AdapterCore | None":
        """Return the first registered adapter matching capability and (optionally) type.

        Args:
            capability (str): Capability name (e.g. `"answerability"`).
            adapter_types (tuple[str, ...] | None): Adapter type strings in
                preference order (e.g. `("alora", "lora")`).  When provided,
                aLoRA is returned before LoRA if both are registered for the same
                capability.  `None` matches any type (insertion order wins).

        Returns:
            _AdapterCore | None: Matching adapter, or `None` if not found.
        """
        # Snapshot into a list: `_added_adapters` is no longer insert-only since
        # `remove_adapter()` (#1528) can delete from it. A concurrent `release()`
        # mutating the dict while this loop holds a live `.values()` view would
        # raise "dictionary changed size during iteration"; iterating a list
        # copy instead is immune to a mutation of the underlying dict.
        #
        # The snapshot also means this lookup can still see an entry that
        # `remove_adapter()` just popped; harmless today because a qualified
        # name is held by either a `LocalFileBinding` or an `IntrinsicAdapter`
        # shim (never both) and the generation path consumes only shims.
        # `remove_adapter()` is public, though, so any registered entry; shim
        # or binding; can be popped: re-check that invariant when #1465 moves
        # generation inside `adapter_scope`.
        adapters = list(getattr(self, "_added_adapters", {}).values())
        # LocalFile/PEFT composed Adapters (Epic #929, issue #1144) don't live
        # in _added_adapters; that dict holds their LocalFileBinding, keyed
        # for the PEFT lifecycle (see LocalHFBackend.add_adapter); only
        # _composed_adapters carries the _AdapterCore object this method
        # matches on. Embedded composed adapters may live in either, per
        # backend (LocalHFBackend uses _composed_adapters; OpenAIBackend
        # stores them directly in _added_adapters), so checking both here
        # keeps this method backend-agnostic without either backend needing
        # to override it.
        adapters += list(getattr(self, "_composed_adapters", {}).values())
        if adapter_types is None:
            for a in adapters:
                if isinstance(a, _AdapterCore) and (
                    a.identity.name == capability or a.identity.capability == capability
                ):
                    return a
            return None
        for preferred_type in adapter_types:
            for a in adapters:
                if (
                    isinstance(a, _AdapterCore)
                    and (
                        a.identity.name == capability
                        or a.identity.capability == capability
                    )
                    and a.identity.adapter_type == preferred_type
                ):
                    return a
        return None


class EmbeddedIntrinsicAdapter(_AdapterCore):
    """Deprecated shim for adapter functions embedded in a Granite Switch model.

    Deprecated:
        Use :class:`~mellea.backends.adapters.Adapter` directly.
        `EmbeddedIntrinsicAdapter` will be removed in a future release
        (Epic #929, issue #1144).

    Unlike PEFT-based adapters that are loaded into the model at runtime,
    embedded adapters are already baked into the model weights and activated
    via control tokens injected by the model's chat template.  Only the I/O
    transformation config (`io.yaml`) is needed; no adapter weights are
    downloaded or loaded.

    Args:
        intrinsic_name (str): Name of the adapter function (e.g. `"answerability"`).
        config (dict): Parsed I/O transformation configuration (from `io.yaml`).
        technology (str): Adapter technology in the switch model; `"lora"` or
            `"alora"`.  Determines where the control token is placed in the
            chat template (beginning of sequence for LoRA, before generation
            prompt for aLoRA).

    Attributes:
        intrinsic_name (str): Name of the adapter function this adapter implements.
        config (dict): Parsed I/O transformation configuration.
        technology (str): `"lora"` or `"alora"`.

    Note:
        `identity`, `io_contract`, and `weights` are internal scaffolding
        populated in `__init__` to satisfy the `Adapter` protocol; they are
        not meaningful consumer-facing attributes.

        - `identity`: always a real value.

        - `io_contract`: the real, declared contract for `intrinsic_name`
          (issue #1516); no longer a placeholder.

        - `weights`: a real `EmbeddedBinding`; activation runs through it.
    """

    def __setattr__(self, name: str, value: object) -> None:
        """Allow mutation; bypasses the frozen restriction on _AdapterCore."""
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        """Allow deletion; bypasses the frozen restriction on _AdapterCore."""
        object.__delattr__(self, name)

    def __init__(self, intrinsic_name: str, config: dict, technology: str = "lora"):
        """Initialize an embedded adapter function with its I/O config."""
        if technology not in ("lora", "alora"):
            raise ValueError(
                f"technology must be 'lora' or 'alora', got '{technology}'"
            )
        warnings.warn(
            "EmbeddedIntrinsicAdapter is deprecated; use Adapter directly (Epic #929, issue #1144).",
            DeprecationWarning,
            stacklevel=2,
        )
        adapter_type = AdapterType.ALORA if technology == "alora" else AdapterType.LORA

        # Old-style Adapter fields; set manually since we no longer inherit from the
        # legacy Adapter ABC.  Preserved for backward compatibility until Phase 4.
        self.name = intrinsic_name
        self.adapter_type = adapter_type
        self.qualified_name = intrinsic_name + "_" + adapter_type.value
        self.backend: Backend | None = None
        self.path: str | None = None

        self.intrinsic_name = intrinsic_name
        self.config = config
        self.technology = technology
        capability = intrinsic_name
        if intrinsic_name in known_intrinsic_names():
            capability = fetch_intrinsic_metadata(intrinsic_name).effective_capability

        # Populate the new Adapter triple so isinstance(self, _AdapterCore) holds.
        # technology is validated above; cast to the Literal type mypy expects.
        identity = Identity(
            name=intrinsic_name,
            adapter_type=cast(Literal["lora", "alora"], technology),
            capability=capability,
        )

        io_contract = get_io_contract(intrinsic_name)

        weights = EmbeddedBinding()

        _AdapterCore.__init__(
            self, identity=identity, io_contract=io_contract, weights=weights
        )

    @staticmethod
    def from_model_directory(
        model_path: str | pathlib.Path, intrinsic_name: str | None = None
    ) -> list["EmbeddedIntrinsicAdapter"]:
        """Load embedded adapters from a Granite Switch model directory.

        Reads `adapter_index.json` and the corresponding `io_configs/*/io.yaml`
        files from the model directory.

        Args:
            model_path (str | pathlib.Path): Path to a Granite Switch model
                directory that contains `adapter_index.json` and `io_configs/`.
            intrinsic_name (str | None): If provided, only load the adapter
                matching this adapter function name. `None` loads all adapters.

        Returns:
            list[EmbeddedIntrinsicAdapter]: One adapter per entry in the index.

        Raises:
            FileNotFoundError: If `adapter_index.json` is missing.
            ValueError: If an `io.yaml` file listed in the index cannot be found
                or if no adapters are found.
        """
        import json as _json

        model_path = pathlib.Path(model_path)
        index_path = model_path / "adapter_index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"No adapter_index.json found at {index_path}")

        with open(index_path, encoding="utf-8") as f:
            index = _json.load(f)

        allowed_root = _io_config_root(model_path)
        adapters: list[EmbeddedIntrinsicAdapter] = []
        for entry in index.get("adapters", []):
            entry_name = entry.get("adapter_name")
            if entry_name is None:
                continue
            if intrinsic_name is not None and entry_name != intrinsic_name:
                continue
            io_config_rel = entry.get("io_config")
            if io_config_rel is None:
                continue

            io_config_path = model_path / io_config_rel
            try:
                io_config_path = io_config_path.resolve(strict=True)
            except (FileNotFoundError, OSError):
                raise ValueError(
                    f"io.yaml for adapter function '{entry_name}' "
                    f"not found at {model_path / io_config_rel}"
                )
            if not io_config_path.is_relative_to(allowed_root):
                raise ValueError(
                    f"io_config path for adapter function '{entry_name}' "
                    f"escapes the model directory: {io_config_path}"
                )

            with open(io_config_path, encoding="utf-8") as f:
                config_dict = yaml.safe_load(f)

            adapters.append(
                EmbeddedIntrinsicAdapter(
                    intrinsic_name=entry_name,
                    config=config_dict,
                    technology=entry.get("technology", "lora"),
                )
            )

        if not adapters:
            if intrinsic_name is not None:
                raise ValueError(
                    f"No adapter found for adapter function '{intrinsic_name}' in {model_path}"
                )
            raise ValueError(f"No adapters found in {model_path}")

        return adapters

    @staticmethod
    def from_hub(
        repo_id: str,
        revision: str = "main",
        cache_dir: str | None = None,
        intrinsic_name: str | None = None,
    ) -> list["EmbeddedIntrinsicAdapter"]:
        """Load embedded adapters from a Granite Switch model on Hugging Face Hub.

        Downloads `adapter_index.json` and the `io_configs/` directory into a
        persistent self-contained local directory, then delegates to
        `from_model_directory`.

        `huggingface_hub.snapshot_download`'s default cache-backed snapshot
        directory populates `io_configs/` with symlinks that resolve into a
        sibling `blobs/` directory *outside* the snapshot root. That breaks the
        contract `from_model_directory` expects (a self-contained model
        directory) and trips its path-escape check. To satisfy that contract,
        the downloaded snapshot is materialised under the Hugging Face cache
        into a self-contained directory keyed by its immutable revision, so
        `io_configs/` contains real files rather than symlinks escaping the
        directory. This preserves standard Hugging Face Hub cache reuse and
        offline loading while preventing stale files from a mutable revision.

        Args:
            repo_id (str): Hugging Face Hub repository ID
                (e.g. `"ibm-granite/granite-switch-micro"`).
            revision (str): Git revision to download from.
            cache_dir (str | None): Local cache directory; `None` for the default.
            intrinsic_name (str | None): If provided, only load the adapter
                matching this adapter function name. `None` loads all adapters.

        Returns:
            list[EmbeddedIntrinsicAdapter]: One adapter per entry in the index.

        Raises:
            ImportError: If `huggingface_hub` is not installed.
            PermissionError: If the repository is private or gated and the
                current Hugging Face credentials do not grant access.
            FileNotFoundError: If the downloaded snapshot has no
                `adapter_index.json` (wrong repo/revision, not a Granite Switch
                model, or a stale cache).
            ValueError: If no adapters are found (delegated from
                `from_model_directory`).
        """
        try:
            import huggingface_hub
            from huggingface_hub.constants import HF_HUB_CACHE
            from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
        except ImportError as e:
            raise ImportError(
                "huggingface_hub is required to download embedded adapter configs from "
                'Hugging Face Hub. Please install it with: pip install "mellea[switch]"'
            ) from e

        try:
            snapshot_root = pathlib.Path(
                huggingface_hub.snapshot_download(
                    repo_id=repo_id,
                    allow_patterns=[
                        "adapter_index.json",
                        "io_configs/**",
                        "chat_template.jinja",
                    ],
                    cache_dir=cache_dir,
                    revision=revision,
                )
            )
        except (GatedRepoError, RepositoryNotFoundError) as e:
            auth_hint = (
                f"Could not access '{repo_id}' on Hugging Face Hub. If this is a "
                "private or gated repository, authenticate first (run "
                "`huggingface-cli login` or set the HF_TOKEN environment variable) "
                "and confirm your account has been granted access to the repository. "
                "Otherwise, the repository ID may be misspelled."
            )
            raise PermissionError(auth_hint) from e

        cache_root = pathlib.Path(cache_dir or HF_HUB_CACHE)
        cache_key = hashlib.sha256(
            f"{repo_id}\0{snapshot_root.name}".encode()
        ).hexdigest()
        local_root = cache_root / "mellea" / "embedded-adapter-configs" / cache_key

        try:
            # Validate the cache by content, not just directory existence: a
            # prior run that crashed or lost a `temporary_dir.replace()` race
            # (see below) can leave `local_root` as a directory missing
            # `adapter_index.json`, which `is_dir()` alone would trust forever.
            if not (local_root / "adapter_index.json").is_file():
                local_root.parent.mkdir(parents=True, exist_ok=True)
                if local_root.is_dir():
                    # Invalid leftover from a prior partial run -- clear it so
                    # `temporary_dir.replace(local_root)` below doesn't fail
                    # trying to rename onto a non-empty stale directory.
                    shutil.rmtree(local_root)
                temporary_dir = pathlib.Path(
                    tempfile.mkdtemp(dir=local_root.parent, prefix=f"{cache_key}-")
                )
                try:
                    import json as _json

                    index_path = snapshot_root / "adapter_index.json"
                    with open(index_path, encoding="utf-8") as f:
                        index = _json.load(f)
                    shutil.copyfile(index_path, temporary_dir / "adapter_index.json")

                    snapshot_cache_root = snapshot_root.parent.parent.resolve()
                    for entry in index.get("adapters", []):
                        io_config_rel = entry.get("io_config")
                        if io_config_rel is None:
                            continue
                        io_config_path = (snapshot_root / io_config_rel).resolve(
                            strict=True
                        )
                        if not io_config_path.is_relative_to(snapshot_cache_root):
                            raise ValueError(
                                f"io_config path '{io_config_rel}' escapes "
                                "the downloaded Hugging Face snapshot"
                            )
                        destination = temporary_dir / io_config_rel
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(io_config_path, destination)

                    adapters = EmbeddedIntrinsicAdapter.from_model_directory(
                        temporary_dir, intrinsic_name=intrinsic_name
                    )
                    try:
                        temporary_dir.replace(local_root)
                    except OSError:
                        if not local_root.is_dir():
                            raise
                    else:
                        return adapters
                finally:
                    if temporary_dir.exists():
                        shutil.rmtree(temporary_dir)

            return EmbeddedIntrinsicAdapter.from_model_directory(
                local_root, intrinsic_name=intrinsic_name
            )
        except FileNotFoundError as e:
            # snapshot_download succeeded but the index is absent: wrong
            # repo/revision, a repo that isn't a Granite Switch model, or a
            # stale cache. Replace the cryptic snapshot-cache path with a
            # repo-scoped message that names authentication as one possible
            # cause without asserting it.
            raise FileNotFoundError(
                f"adapter_index.json was not found in the downloaded snapshot of "
                f"'{repo_id}'. Verify it is a Granite Switch model and that the "
                f"revision '{revision}' is correct; if the repository is private "
                "or gated, confirm you are authenticated (run "
                "`huggingface-cli login` or set the HF_TOKEN environment variable)."
            ) from e
        except ValueError as e:
            if intrinsic_name is not None:
                raise ValueError(
                    f"No adapter found for adapter function '{intrinsic_name}' in {repo_id}"
                ) from e
            raise ValueError(f"No adapters found in {repo_id}") from e

    @staticmethod
    def from_source(
        source: str,
        revision: str = "main",
        cache_dir: str | None = None,
        intrinsic_name: str | None = None,
    ) -> list["EmbeddedIntrinsicAdapter"]:
        """Load embedded adapters from a local directory or Hugging Face Hub.

        Automatically detects whether `source` is a local filesystem path
        or a Hugging Face Hub repo ID, and delegates accordingly.

        Args:
            source (str): Local path to a model directory, or a Hugging Face
                Hub repo ID (e.g. `"ibm-granite/granite-switch-micro"`).
            revision (str): Git revision (only used for Hub downloads).
            cache_dir (str | None): Cache directory (only used for Hub downloads).
            intrinsic_name (str | None): If provided, only load the adapter
                matching this adapter function name. `None` loads all adapters.

        Returns:
            list[EmbeddedIntrinsicAdapter]: One adapter per entry in the index.
        """
        if pathlib.Path(source).is_dir():
            return EmbeddedIntrinsicAdapter.from_model_directory(
                source, intrinsic_name=intrinsic_name
            )
        return EmbeddedIntrinsicAdapter.from_hub(
            source,
            revision=revision,
            cache_dir=cache_dir,
            intrinsic_name=intrinsic_name,
        )


class CustomIntrinsicAdapter(IntrinsicAdapter):
    """Deprecated shim for user-defined custom adapter functions.

    .. deprecated::
        Use :class:`~mellea.backends.adapters.Adapter` directly.
        `CustomIntrinsicAdapter` will be removed in a future release
        (Epic #929, issue #1144).

    This class has the same functionality as `IntrinsicAdapter`, except that
    its constructor monkey-patches Mellea global variables to enable the backend
    to load the user's adapter.

    Args:
        model_id (str): The Hugging Face model ID used for downloading model weights;
            expected format is `"<user-id>/<repo-name>"`.
        intrinsic_name (str | None): Catalog name for the adapter function; defaults to the
            repository name portion of `model_id` if not provided.
        base_model_name (str): The short name of the base model (NOT its repo ID).
    """

    def __init__(
        self, *, model_id: str, intrinsic_name: str | None = None, base_model_name: str
    ):
        """Initialize CustomIntrinsicAdapter and patch the global adapter function catalog if needed."""
        warnings.warn(
            "CustomIntrinsicAdapter is deprecated; use Adapter directly (Epic #929, issue #1144).",
            DeprecationWarning,
            stacklevel=2,
        )
        assert re.match(".*/.*", model_id), (
            "expected a Hugging Face model id with format <user-id>/<repo-name>"
        )
        intrinsic_name = (
            intrinsic_name if intrinsic_name is not None else model_id.split("/")[1]
        )

        # patch the catalog. TODO this is a temporary hack until we re-org adapters.
        from mellea.backends.adapters import catalog

        if intrinsic_name not in catalog._INTRINSICS_CATALOG:
            catalog._INTRINSICS_CATALOG_ENTRIES.append(
                catalog.IntrinsicsCatalogEntry(
                    name=intrinsic_name, repo_id=model_id, revision="main"
                )
            )
            catalog._INTRINSICS_CATALOG = {
                e.name: e for e in catalog._INTRINSICS_CATALOG_ENTRIES
            }

        # Suppress DeprecationWarning from the IntrinsicAdapter shim: the warning we
        # emitted above is already correctly attributed to the caller's frame.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            IntrinsicAdapter.__init__(
                self, intrinsic_name=intrinsic_name, base_model_name=base_model_name
            )
