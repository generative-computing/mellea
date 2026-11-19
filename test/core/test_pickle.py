# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for pickling core Mellea objects (issue #1535).

Pickling a `ModelOutputThunk` must drop the in-flight generation machinery
(`_gen`) and backend-native state (`raw.response`, `generation.logits`), mirroring
the existing `__copy__`/`__deepcopy__` contract: a reloaded thunk is a computed,
post-generation object with no live asyncio tasks, queues, or backend callbacks.
"""

import asyncio
import pickle

import pytest

from mellea import start_session
from mellea.backends.model_ids import IBM_GRANITE_4_2_3B
from mellea.core import CBlock, ComputedModelOutputThunk, ModelOutputThunk
from mellea.core.base import GenerateType
from mellea.stdlib.context.chat import ChatContext


def _computed_thunk() -> ModelOutputThunk:
    """A fully-populated computed thunk with metadata, tool calls elided."""
    mot = ModelOutputThunk("hello world")
    mot.thinking = "some reasoning"
    mot.generation.usage = {
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "total_tokens": 5,
    }
    mot.generation.model = "test-model"
    mot.generation.provider = "test-provider"
    mot._meta = {"key": "value"}
    return mot


def test_computed_mot_pickle_roundtrip():
    """A computed thunk round-trips its output and portable metadata."""
    mot = _computed_thunk()

    loaded = pickle.loads(pickle.dumps(mot))

    assert loaded.value == "hello world"
    assert loaded.is_computed()
    assert loaded.thinking == "some reasoning"
    assert loaded._meta == {"key": "value"}
    assert loaded.generation.usage == {
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "total_tokens": 5,
    }
    assert loaded.generation.model == "test-model"
    assert loaded.generation.provider == "test-provider"


def test_mot_with_async_task_in_gen_pickles():
    """A thunk carrying a live asyncio Task pickles; the loaded `_gen` is fresh."""

    async def build_and_pickle() -> ModelOutputThunk:
        mot = _computed_thunk()

        async def _noop() -> None:
            await asyncio.sleep(10)

        async def _process(thunk: ModelOutputThunk, chunk: object) -> None:
            return None

        async def _post_process(thunk: ModelOutputThunk) -> None:
            return None

        task = asyncio.ensure_future(_noop())
        mot._gen.generate = task
        mot._gen.generate_type = GenerateType.ASYNC
        mot._gen.process = _process
        mot._gen.post_process = _post_process
        mot._gen.cancel_hook = lambda: None
        mot._gen.queue.put_nowait("buffered-chunk")
        try:
            return pickle.loads(pickle.dumps(mot))
        finally:
            task.cancel()

    loaded = asyncio.run(build_and_pickle())

    # Output survives.
    assert loaded.value == "hello world"
    assert loaded.is_computed()
    # In-flight machinery is reset to a fresh, non-generating state.
    assert loaded._gen.generate is None
    assert loaded._gen.generate_extra is None
    assert loaded._gen.generate_type is GenerateType.NONE
    assert loaded._gen.process is None
    assert loaded._gen.post_process is None
    assert loaded._gen.on_computed is None
    assert loaded._gen.cancel_hook is None
    assert loaded._gen.queue.empty()


class _Unpicklable:
    """A sentinel that raises if pickle tries to serialize it."""

    def __reduce__(self):
        raise RuntimeError("this object must not be pickled")


def test_mot_drops_raw_and_logits():
    """Backend-native `raw.response` and torch logits are dropped, not pickled."""
    mot = _computed_thunk()
    mot.raw.provider = "test-provider"
    mot.raw.response = _Unpicklable()
    mot.raw.streamed_chunks = [_Unpicklable(), _Unpicklable()]
    mot.generation.logits = (_Unpicklable(),)  # type: ignore[assignment]
    mot.generation.raw_logits = (_Unpicklable(),)  # type: ignore[assignment]

    loaded = pickle.loads(pickle.dumps(mot))

    # The non-portable payloads are gone...
    assert loaded.raw.response is None
    assert loaded.raw.streamed_chunks is None
    assert loaded.generation.logits is None
    assert loaded.generation.raw_logits is None
    # ...but the portable sibling fields are preserved.
    assert loaded.raw.provider == "test-provider"
    assert loaded.generation.provider == "test-provider"


def test_mot_self_referential_parsed_repr_roundtrips():
    """A thunk whose `parsed_repr` points at itself survives as a self-cycle."""
    mot = _computed_thunk()
    mot.parsed_repr = mot  # parsing that returns the thunk unchanged

    loaded = pickle.loads(pickle.dumps(mot))

    assert loaded.parsed_repr is loaded
    assert loaded.value == "hello world"


def test_pickle_does_not_mutate_source():
    """Pickling drops backend-native fields on the copy, not on the source thunk."""
    mot = _computed_thunk()
    sentinel = _Unpicklable()
    mot.raw.response = sentinel
    mot.generation.logits = (sentinel,)  # type: ignore[assignment]

    pickle.dumps(mot)

    # The original thunk keeps its backend-native payloads.
    assert mot.raw.response is sentinel
    assert mot.generation.logits == (sentinel,)


def test_uncomputed_mot_pickle_raises():
    """Pickling an uncomputed thunk raises, mirroring `__copy__`."""
    mot = ModelOutputThunk(None)
    assert not mot.is_computed()

    with pytest.raises(RuntimeError, match="uncomputed"):
        pickle.dumps(mot)


def test_computed_mot_subclass_pickle_preserves_type():
    """A `ComputedModelOutputThunk` round-trips as the same class, invariant intact."""
    mot = ComputedModelOutputThunk(_computed_thunk())

    loaded = pickle.loads(pickle.dumps(mot))

    assert type(loaded) is ComputedModelOutputThunk
    assert loaded.value == "hello world"
    assert loaded.is_computed()
    # The computed invariant still rejects an uncomputing write.
    with pytest.raises(AttributeError):
        loaded.value = None  # type: ignore[assignment]


@pytest.mark.e2e
@pytest.mark.ollama
def test_backend_produced_context_pickle_drops_raw_payloads():
    """A context holding a real Ollama generation pickles without backend-native payloads."""
    m = start_session("ollama", model_id=IBM_GRANITE_4_2_3B, ctx=ChatContext())
    m.instruct("Say hello in one word.")

    # Sanity: the backend populated the SDK payload on both the raw and the log.
    thunks = [i for i in m.ctx.as_list() if isinstance(i, ModelOutputThunk)]
    assert thunks, "expected the generated thunk in the context"
    source = thunks[-1]
    assert source.raw.response is not None
    assert source._generate_log is not None
    assert source._generate_log.model_output is not None

    loaded_ctx = pickle.loads(pickle.dumps(m.ctx))

    loaded_thunks = [i for i in loaded_ctx.as_list() if isinstance(i, ModelOutputThunk)]
    loaded = loaded_thunks[-1]
    assert loaded.value == source.value
    assert loaded.raw.response is None
    assert loaded.raw.streamed_chunks is None
    assert loaded._generate_log is not None
    assert loaded._generate_log.model_output is None
    # The source keeps its SDK payload.
    assert source.raw.response is not None


def test_cblock_pickle_roundtrip():
    """A `CBlock` round-trips its value and metadata."""
    cb = CBlock("some content", meta={"a": 1})

    loaded = pickle.loads(pickle.dumps(cb))

    assert loaded.value == "some content"
    assert loaded._meta == {"a": 1}


def test_chatcontext_pickle_roundtrip():
    """A `ChatContext` with content and a window compactor round-trips its history."""
    ctx = ChatContext(window_size=5).add(_computed_thunk()).add(CBlock("a follow-up"))

    loaded = pickle.loads(pickle.dumps(ctx))

    original_items = ctx.as_list()
    loaded_items = loaded.as_list()
    assert len(loaded_items) == len(original_items) == 2
    assert loaded_items[0].value == "hello world"
    assert loaded_items[1].value == "a follow-up"
