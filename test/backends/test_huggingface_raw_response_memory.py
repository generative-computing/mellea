# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retention and fidelity tests for `mot.raw.response` on the HF backend.

Two code paths are covered:

**Raw batch path** (`LocalHFBackend._generate_from_raw`):
`_generate_from_raw` populates `mot.raw.response` with a per-row slice of the
batch output. Tests pin the invariants that make that safe:

- a MOT retains only its own row, never the whole batch allocation;
- once the caller holds nothing but MOTs, the batch tensors are actually freed;
- a CPU/MPS run pays neither a full GC nor a CUDA allocator flush;
- `raw.response` never misreports the type or the contents of what `generate()`
  actually returned.

**Chat / post-processing path** (`LocalHFBackend.post_processing`):

- when caching is enabled, cleared fields (scores, logits) release their tensors
  even though the `GenerateDecoderOnlyOutput` container is kept alive;
- when caching is disabled, `raw.response` is set to `None` and every tensor
  reachable through the HF output graph (sequences, scores, logits,
  past_key_values) is released.

Sizes here are tiny; the assertions are on *which* storage is retained, not on
how much memory a real model would use.
"""

import copy
import gc
import logging
import weakref
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

from mellea.core.base import Component

if TYPE_CHECKING:
    import torch as _torch

import pytest

torch = pytest.importorskip("torch", reason="torch not installed — install mellea[hf]")
pytest.importorskip(
    "transformers", reason="transformers not installed — install mellea[hf]"
)
pytest.importorskip(
    "llguidance", reason="llguidance not installed — install mellea[hf]"
)

from transformers.cache_utils import CacheLayerMixin, DynamicCache
from transformers.generation.utils import (
    GenerateBeamDecoderOnlyOutput,
    GenerateDecoderOnlyOutput,
)

from mellea.backends import ModelOption
from mellea.backends.huggingface import LocalHFBackend
from mellea.core import ModelOutputThunk
from mellea.stdlib.components import Message

_VOCAB = 64
_PROMPT_LEN = 1
_SEQ_LEN = 4


def _make_backend(batch_size: int) -> LocalHFBackend:
    """A CPU-only LocalHFBackend whose tokenizer returns a fixed batch encoding."""
    mock_tok = MagicMock(eos_token_id=0, vocab_size=_VOCAB)
    mock_tok._tokenizer = MagicMock()
    mock_tok._tokenizer.get_vocab_size.return_value = _VOCAB
    mock_tok.__len__ = MagicMock(return_value=_VOCAB)
    mock_model = MagicMock(vocab_size=_VOCAB)
    with (
        patch("mellea.backends.huggingface.llguidance") as mock_llg,
        patch("mellea.backends.huggingface.set_seed"),
    ):
        mock_llg.hf.from_tokenizer.return_value = MagicMock(vocab_size=_VOCAB)
        backend = LocalHFBackend(
            model_id="ibm-granite/granite-3.3-8b-instruct",
            custom_config=(mock_tok, mock_model, torch.device("cpu")),
        )

    input_ids = torch.zeros(batch_size, _PROMPT_LEN, dtype=torch.long)
    encoding = MagicMock()
    encoding.__getitem__ = lambda _self, k: (
        input_ids
        if k == "input_ids"
        else torch.ones(batch_size, _PROMPT_LEN, dtype=torch.long)
    )
    encoding.to = MagicMock(return_value=encoding)
    backend._tokenizer = MagicMock(eos_token_id=0, vocab_size=_VOCAB)
    backend._tokenizer.__len__ = MagicMock(return_value=_VOCAB)
    backend._tokenizer.return_value = encoding
    # A plain function, not a MagicMock: `batch_decode` is handed slices of the batch
    # and a mock would record them in `call_args`, keeping the batch alive and
    # defeating the retention tests below.
    decoded = [f"result_{i}" for i in range(batch_size)]

    def _batch_decode(_sequences: Any, **_kwargs: Any) -> list[str]:
        return decoded

    backend._tokenizer.batch_decode = _batch_decode
    return backend


def _batch_output(
    batch_size: int, *, n_steps: int = 0, raw_logits: bool = False
) -> GenerateDecoderOnlyOutput:
    """A batch `generate()` result: `(batch_size, _SEQ_LEN)` sequences plus optional scores."""
    step_shape = (batch_size, _VOCAB)
    return GenerateDecoderOnlyOutput(
        sequences=torch.zeros(batch_size, _SEQ_LEN, dtype=torch.long),
        scores=tuple(torch.zeros(*step_shape) for _ in range(n_steps)) or None,
        logits=(tuple(torch.ones(*step_shape) for _ in range(n_steps)) or None)
        if raw_logits
        else None,
        attentions=None,
        hidden_states=None,
        past_key_values=None,
    )


async def _drive(
    backend: LocalHFBackend,
    outputs: Any,
    batch_size: int,
    model_options: dict | None = None,
) -> list[ModelOutputThunk]:
    """Run `generate_from_raw` with `model.generate` stubbed to return `outputs`.

    The stub is a plain async function rather than an `AsyncMock` so that no mock
    call record keeps `outputs` alive after the call returns — the retention tests
    below depend on this frame being the only strong reference.
    """

    async def _fake_to_thread(*_args: Any, **_kwargs: Any) -> Any:
        return outputs

    actions: list[Component[Any]] = [
        Message("user", f"prompt {i}") for i in range(batch_size)
    ]
    with (
        patch("mellea.backends.huggingface.asyncio.to_thread", new=_fake_to_thread),
        patch.object(backend, "do_generate_walks", new=AsyncMock()),
        patch.object(backend, "formatter") as mock_fmt,
    ):
        mock_fmt.print = MagicMock(return_value="prompt")
        return await backend.generate_from_raw(
            actions, MagicMock(), model_options=model_options or {}
        )


def _row_bytes(dtype: "_torch.dtype", length: int) -> int:
    return length * torch.empty(0, dtype=dtype).element_size()


# --- retention: a MOT must keep its own row, not the batch --------------------


async def test_raw_response_sequences_retain_only_their_own_row():
    """`raw.response.sequences` must not keep the whole batch allocation alive.

    A `t[i:i+1]` view shares `t`'s storage, so every MOT would pin the full
    `(batch, seq_len)` buffer for its entire lifetime. Slices handed to a MOT must
    own compact storage — the same reason `generation.logits` is cloned.
    """
    batch_size = 8
    backend = _make_backend(batch_size)
    outputs = _batch_output(batch_size)
    batch_bytes = outputs.sequences.untyped_storage().nbytes()
    row_bytes = _row_bytes(torch.long, _SEQ_LEN)
    assert batch_bytes > row_bytes, "test setup: batch must be larger than one row"

    results = await _drive(backend, outputs, batch_size)

    for i, result in enumerate(results):
        held = result.raw.response.sequences.untyped_storage().nbytes()
        assert held == row_bytes, (
            f"item {i}: raw.response.sequences retains {held} bytes of storage but its "
            f"own row is only {row_bytes} bytes — it is a view pinning the whole "
            f"{batch_bytes}-byte batch"
        )


async def test_raw_response_scores_retain_only_their_own_row():
    """`raw.response.scores` must not keep the whole batch scores tensors alive.

    Scores are the expensive field: one `(batch, vocab)` tensor per generated
    token. A view into each pins every row of every step.
    """
    batch_size, n_steps = 8, 3
    backend = _make_backend(batch_size)
    outputs = _batch_output(batch_size, n_steps=n_steps)
    batch_bytes = outputs.scores[0].untyped_storage().nbytes()
    row_bytes = _row_bytes(torch.float32, _VOCAB)

    results = await _drive(backend, outputs, batch_size)

    for i, result in enumerate(results):
        assert result.raw.response.scores is not None
        for step, tensor in enumerate(result.raw.response.scores):
            held = tensor.untyped_storage().nbytes()
            assert held == row_bytes, (
                f"item {i} step {step}: raw.response.scores retains {held} bytes but its "
                f"own row is {row_bytes} bytes — it is a view pinning the whole "
                f"{batch_bytes}-byte step tensor"
            )


async def _run_holding_only_weakrefs(
    backend: LocalHFBackend, batch_size: int, n_steps: int
) -> tuple[list[ModelOutputThunk], dict[str, Any]]:
    """Run `generate_from_raw` and return the MOTs plus weakrefs to the batch tensors.

    Every strong reference to the batch (the tensors and the `GenerateDecoderOnlyOutput`
    holding them) lives in this frame, so all of them are gone once it returns.
    Anything still keeping the batch alive after that is held by the returned MOTs.
    """
    outputs = _batch_output(batch_size, n_steps=n_steps)
    assert outputs.scores is not None
    refs = {
        "sequences": weakref.ref(outputs.sequences),
        "scores": [weakref.ref(s) for s in outputs.scores],
    }
    results = await _drive(backend, outputs, batch_size)
    del outputs
    return results, refs


async def test_batch_tensors_are_freed_once_only_mots_are_held():
    """Holding the returned MOTs must not keep the batch tensors alive.

    This is the invariant that matters in practice: a caller that keeps one MOT out
    of a 32-prompt batch should not be pinning all 32 rows of sequences and of every
    step's scores. The MOTs must still carry their own usable data afterwards.
    """
    batch_size, n_steps = 4, 2
    backend = _make_backend(batch_size)

    results, refs = await _run_holding_only_weakrefs(backend, batch_size, n_steps)
    gc.collect()
    gc.collect()  # second pass: release anything freed by breaking a cycle

    assert refs["sequences"]() is None, (
        "batch sequences tensor is still alive while only MOTs are held — "
        "raw.response.sequences is a view into it"
    )
    for step, ref in enumerate(refs["scores"]):
        assert ref() is None, (
            f"batch scores tensor for step {step} is still alive while only MOTs are "
            "held — raw.response.scores holds views into it"
        )

    # The data must survive the batch being freed.
    for i, result in enumerate(results):
        assert result.raw.response.sequences.shape == (1, _SEQ_LEN), (
            f"item {i}: sequences must remain usable after the batch is released"
        )
        assert len(result.raw.response.scores) == n_steps


async def test_deepcopy_of_result_does_not_duplicate_the_batch():
    """`copy.deepcopy(mot)` must not allocate a whole extra batch.

    `torch.Tensor.__deepcopy__` deep-copies the *storage* and rebuilds the tensor
    with the original size/stride/offset, so deep-copying a row view reallocates the
    entire batch buffer. Asserting only that the data pointers differ hides this.
    """
    batch_size = 8
    backend = _make_backend(batch_size)
    outputs = _batch_output(batch_size)
    batch_bytes = outputs.sequences.untyped_storage().nbytes()
    row_bytes = _row_bytes(torch.long, _SEQ_LEN)

    results = await _drive(backend, outputs, batch_size)
    deep = copy.deepcopy(results[0])

    held = deep.raw.response.sequences.untyped_storage().nbytes()
    assert held == row_bytes, (
        f"deepcopy allocated {held} bytes for a {row_bytes}-byte row — it duplicated "
        f"the whole {batch_bytes}-byte batch because the slice was a view"
    )


# --- cleanup cost -------------------------------------------------------------


async def test_generate_from_raw_does_not_force_gc_or_cuda_flush_without_cuda():
    """No full GC and no CUDA allocator flush on a CPU/MPS run.

    `gc.collect()` is a full generational collection and `torch.cuda.empty_cache()`
    returns pooled blocks to the driver, making the next `generate()` pay fresh
    allocations. Neither belongs on the per-call path, least of all when there is no
    CUDA device at all.
    """
    batch_size = 2
    backend = _make_backend(batch_size)
    outputs = _batch_output(batch_size, n_steps=1)

    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.cuda.empty_cache") as mock_empty_cache,
        patch("gc.collect") as mock_collect,
    ):
        await _drive(backend, outputs, batch_size)

    assert mock_collect.call_count == 0, (
        f"generate_from_raw forced {mock_collect.call_count} full GC pass(es) on a "
        "non-CUDA device"
    )
    assert mock_empty_cache.call_count == 0, (
        f"generate_from_raw called torch.cuda.empty_cache() {mock_empty_cache.call_count} "
        "time(s) with no CUDA device available"
    )


# --- fidelity: raw.response must describe what generate() returned ------------


async def test_raw_response_preserves_beam_search_output_type_and_fields():
    """A beam-search batch must not be reported as a plain decoder-only output.

    With `num_beams > 1`, `generate()` returns `GenerateBeamDecoderOnlyOutput` with
    `sequences_scores`/`beam_indices`, and `scores` has a leading dimension of
    `batch * num_beams` — so per-item row indexing is wrong as well. Either mirror
    the real type (carrying the beam fields) or leave `raw.response` unset; silently
    relabelling it as `GenerateDecoderOnlyOutput` is the one unacceptable option.
    """
    batch_size, num_beams, n_steps = 2, 3, 2
    backend = _make_backend(batch_size)
    outputs = GenerateBeamDecoderOnlyOutput(
        sequences=torch.zeros(batch_size, _SEQ_LEN, dtype=torch.long),
        sequences_scores=torch.zeros(batch_size),
        scores=tuple(
            torch.zeros(batch_size * num_beams, _VOCAB) for _ in range(n_steps)
        ),
        logits=None,
        beam_indices=torch.zeros(batch_size, _SEQ_LEN, dtype=torch.long),
        attentions=None,
        hidden_states=None,
        past_key_values=None,
    )

    results = await _drive(
        backend, outputs, batch_size, model_options={"num_beams": num_beams}
    )

    for i, result in enumerate(results):
        raw = result.raw.response
        if raw is None:
            continue  # opting out is acceptable; misreporting is not
        assert isinstance(raw, GenerateBeamDecoderOnlyOutput), (
            f"item {i}: generate() returned GenerateBeamDecoderOnlyOutput but "
            f"raw.response is {type(raw).__name__}"
        )
        assert raw.sequences_scores is not None, (
            f"item {i}: beam sequences_scores was dropped"
        )
        assert raw.beam_indices is not None, f"item {i}: beam_indices was dropped"


async def test_raw_response_is_never_emitted_with_null_sequences():
    """`raw.response` must never be a sequence-bearing output whose sequences are None.

    The raw path guards its slice with `isinstance(outputs.sequences, torch.Tensor)`
    and falls back to `None`, casting it to `torch.LongTensor`. Decoding only needs
    `sequences` to be indexable, so a non-tensor batch still produces MOTs with
    values — and a `raw.response` that lies about having sequences. Either drop the
    unreachable branch or leave `raw.response` unset.
    """
    batch_size = 2
    backend = _make_backend(batch_size)
    outputs = GenerateDecoderOnlyOutput(
        sequences=[torch.zeros(_SEQ_LEN, dtype=torch.long) for _ in range(batch_size)],
        scores=None,
        logits=None,
        attentions=None,
        hidden_states=None,
        past_key_values=None,
    )

    results = await _drive(backend, outputs, batch_size)

    for i, result in enumerate(results):
        if result.raw.response is None:
            continue
        assert result.raw.response.sequences is not None, (
            f"item {i}: raw.response was populated with sequences=None"
        )


async def test_raw_response_scores_follow_generate_output_not_the_logits_option():
    """`raw.response.scores` mirrors `generate()`, independent of `ModelOption.LOGITS`.

    The implementation gates on `outputs.scores is not None`, never on the option, so
    scores requested through a passthrough option (or any other route) still land in
    `raw.response`. This is the correct behaviour — the point of the test is that
    "scores is None when LOGITS is not requested" is not the invariant.
    """
    batch_size, n_steps = 2, 2
    backend = _make_backend(batch_size)
    outputs = _batch_output(batch_size, n_steps=n_steps)

    results = await _drive(backend, outputs, batch_size, model_options={})

    for i, result in enumerate(results):
        assert result.raw.response.scores is not None, (
            f"item {i}: generate() returned scores, so raw.response.scores must be set "
            "even though ModelOption.LOGITS was not requested"
        )
        assert len(result.raw.response.scores) == n_steps
        assert result.generation.logits is None, (
            f"item {i}: generation.logits is the option-gated field and must stay unset"
        )


async def test_raw_response_raw_logits_retain_only_their_own_row():
    """`raw.response.logits` gets the same treatment as `scores`.

    Covers the `outputs.logits` branch, which the PR's tests never exercise.
    """
    batch_size, n_steps = 4, 2
    backend = _make_backend(batch_size)
    outputs = _batch_output(batch_size, n_steps=n_steps, raw_logits=True)
    row_bytes = _row_bytes(torch.float32, _VOCAB)

    results = await _drive(
        backend, outputs, batch_size, model_options={ModelOption.RAW_LOGITS: True}
    )

    for i, result in enumerate(results):
        assert result.raw.response.logits is not None
        for step, tensor in enumerate(result.raw.response.logits):
            assert tensor.shape == (1, _VOCAB)
            held = tensor.untyped_storage().nbytes()
            assert held == row_bytes, (
                f"item {i} step {step}: raw.response.logits retains {held} bytes for a "
                f"{row_bytes}-byte row"
            )


# --- one-time notice ----------------------------------------------------------


async def test_omitted_fields_notice_is_logged_once_per_backend(caplog):
    """The "fields omitted" notice fires once per backend, not once per item or call."""
    batch_size = 3
    backend = _make_backend(batch_size)
    needle = "are not available on the raw batch path"

    with caplog.at_level(logging.DEBUG, logger="mellea"):
        await _drive(backend, _batch_output(batch_size), batch_size)
        await _drive(backend, _batch_output(batch_size), batch_size)

    hits = [r for r in caplog.records if needle in r.getMessage()]
    assert len(hits) == 1, (
        f"expected the omitted-fields notice once per backend, saw {len(hits)} across "
        f"two calls of {batch_size} items each"
    )


# --- chat path: the same "null it out" idiom, same defect ---------------------


def _make_kv_cache() -> DynamicCache:
    """Return a small `DynamicCache` with real tensors for weakref testing."""
    cache = DynamicCache()
    cache.update(
        key_states=torch.zeros(1, 1, 1, 4),
        value_states=torch.zeros(1, 1, 1, 4),
        layer_idx=0,
    )
    return cache


async def _post_process_holding_only_weakrefs(
    backend: LocalHFBackend, n_steps: int, *, retain_output: bool = False
) -> tuple[ModelOutputThunk, dict[str, Any], GenerateDecoderOnlyOutput | None]:
    """Run `post_processing` and return the MOT plus weakrefs to the HF output tensors."""
    kv_cache = _make_kv_cache()
    sequences = torch.zeros(1, _SEQ_LEN, dtype=torch.long)
    hf_out = GenerateDecoderOnlyOutput(
        sequences=sequences,
        scores=tuple(torch.zeros(1, _VOCAB) for _ in range(n_steps)),
        logits=tuple(torch.ones(1, _VOCAB) for _ in range(n_steps)),
        attentions=None,
        hidden_states=None,
        past_key_values=kv_cache,
    )
    assert hf_out.logits is not None
    assert hf_out.scores is not None
    assert kv_cache.layers
    assert all(isinstance(layer, CacheLayerMixin) for layer in kv_cache.layers), (
        "test setup: expected standard KV cache layers"
    )

    refs = {
        "container": weakref.ref(hf_out),
        "sequences": [weakref.ref(sequences)],
        "logits": [weakref.ref(t) for t in hf_out.logits],
        "scores": [weakref.ref(t) for t in hf_out.scores],
        "past_key_values": [
            weakref.ref(t)
            for layer in kv_cache.layers
            if isinstance(layer, CacheLayerMixin)
            for t in (layer.keys, layer.values)
            if t is not None
        ],
    }
    del kv_cache, sequences

    mot = ModelOutputThunk(value="hi")
    mot._call.action = Message("user", "noop")
    mot._call.model_options = {}
    mot.raw.response = hf_out
    retained = hf_out if retain_output else None
    del hf_out

    await backend.post_processing(
        mot, [], None, False, {}, None, torch.zeros(1, _PROMPT_LEN, dtype=torch.long)
    )
    return mot, refs, retained


async def test_post_processing_clearing_raw_logits_actually_releases_them():
    """Clearing raw scores and logits must release their tensors.

    `GenerateDecoderOnlyOutput` is a `ModelOutput`, i.e. an `OrderedDict`
    subclass that mirrors fields into the mapping. Clearing an attribute alone
    can leave the mapping entry—and therefore its tensors—alive. When
    `raw.response` is retained, post-processing must remove all references to
    the cleared scores and logits.
    """
    backend = _make_backend(1)
    backend._use_caches = True  # keeps raw.response, so the container survives

    mot, refs, _ = await _post_process_holding_only_weakrefs(backend, n_steps=2)
    gc.collect()
    gc.collect()

    assert mot.raw.response is not None, "test setup: raw.response should be retained"

    for field in ("scores", "logits"):
        assert getattr(mot.raw.response, field) is None, (
            f"test setup: {field} attribute was not cleared"
        )

        for step, ref in enumerate(refs[field]):
            assert ref() is None, (
                f"raw {field} tensor for step {step} is still alive after "
                f"hf_output.{field} was cleared"
            )

    # The retained ModelOutput must remain a valid OrderedDict after cleanup.
    # `dict.__delitem__` bypasses OrderedDict bookkeeping and corrupts these
    # mapping operations even though the tensor references are released.
    assert tuple(mot.raw.response.keys()) == ("sequences",)
    assert [key for key, _value in mot.raw.response.items()] == ["sequences"]

    deep = copy.deepcopy(mot.raw.response)
    assert tuple(deep.keys()) == ("sequences",)
    assert [key for key, _value in deep.items()] == ["sequences"]
    assert deep.sequences is not None


async def test_post_processing_clears_all_tensor_fields_when_caching_disabled():
    """When `_use_caches=False`, `post_processing` clears tensor-bearing fields.

    The HF output `sequences`, `scores`, `logits`, and `past_key_values`
    must be released after post_processing.
    """
    backend = _make_backend(1)
    backend._use_caches = False  # triggers the no-caching clearing path

    mot, refs, retained = await _post_process_holding_only_weakrefs(
        backend, n_steps=2, retain_output=True
    )

    gc.collect()
    gc.collect()

    assert mot.raw.response is None, (
        "raw.response should be dropped on the no-caching path"
    )
    assert retained is not None, "test setup: retain the output container"

    for field in ("sequences", "scores", "logits", "past_key_values"):
        for i, ref in enumerate(refs[field]):
            assert ref() is None, (
                f"{field} tensor {i} is still alive after post_processing "
                "with _use_caches=False"
            )
    # Even though raw.response was dropped, another caller may still retain the
    # ModelOutput. The no-cache cleanup must not corrupt its OrderedDict state.
    assert tuple(retained.keys()) == ()
    assert list(retained.items()) == []

    deep = copy.deepcopy(retained)
    assert tuple(deep.keys()) == ()
    assert list(deep.items()) == []
    assert deep.sequences is None
    assert deep.scores is None
    assert deep.logits is None
    assert deep.past_key_values is None
