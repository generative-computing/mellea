import asyncio
from copy import copy
import faulthandler
import random
import time
from typing import Any, Coroutine
from unittest.mock import Mock

import pytest
import torch

from mellea import MelleaSession
from mellea.backends.adapters.adapter import GraniteCommonAdapter
from mellea.backends.cache import SimpleLRUCache
from mellea.backends.formatter import TemplateFormatter
from mellea.backends.huggingface import HFGenerationLock, LocalHFBackend, _assert_correct_adapters
from mellea.backends.types import ModelOption
from mellea.stdlib.base import CBlock, ChatContext, Context, GenerateType, ModelOutputThunk
from mellea.stdlib.chat import Message
from mellea.stdlib.intrinsics.intrinsic import Intrinsic


@pytest.fixture(scope="module")
def backend():
    """Shared HuggingFace backend for all tests in this module."""
    backend = LocalHFBackend(
        model_id="ibm-granite/granite-3.3-8b-instruct",
        formatter=TemplateFormatter(model_id="ibm-granite/granite-4.0-tiny-preview"),
        cache=SimpleLRUCache(5),
    )
    backend.add_adapter(
        GraniteCommonAdapter(
            "requirement_check", base_model_name=backend.base_model_name
        )
    )
    backend.add_adapter(
        GraniteCommonAdapter(
            "answerability", base_model_name=backend.base_model_name
        )
    )
    return backend


@pytest.fixture(scope="function")
def session(backend):
    """Fresh HuggingFace session for each test."""
    session = MelleaSession(backend, ctx=ChatContext())
    yield session
    session.reset()

@pytest.mark.qualitative
async def test_generate_with_lock(backend):
    # Enable the faulthandler for this test.
    faulthandler.enable(all_threads=True)

    # Create local versions of these objects so that mocking
    # doesn't impact other functions. Don't do this in regular code,
    # the copying is complex.
    b: LocalHFBackend = copy(backend)
    model = copy(b._model)
    b._model = model
    b._added_adapters = {}
    b._loaded_adapters = {}
    b._generate_lock = HFGenerationLock(b)
    b.add_adapter(
        GraniteCommonAdapter(
            "requirement_check", base_model_name=b.base_model_name
        )
    )
    b.add_adapter(
        GraniteCommonAdapter(
            "answerability", base_model_name=b.base_model_name
        )
    )

    memoized = dict()
    gen_func = model.generate
    def mock_func(input_ids, *args, **kwargs):
        """Mocks the generate function. Must call `populate_mocked_dict` with each input that must be cached before using this."""
        for key, val in memoized.items():
            if torch.equal(key, input_ids):
                time.sleep(random.uniform(.1, .5)) # Simulate a bit of work.
                return val
        assert False, "did not get a cached response"

    # Safely create the dict.
    def populate_mocked_dict(input_ids, *args, **kwargs):
        """Generates the model output and adds to the memoized dict."""
        output = gen_func(input_ids, *args, **kwargs)  # type: ignore
        memoized[input_ids] = output
        return output

    model.generate = Mock(side_effect=populate_mocked_dict)
    assert not isinstance(backend._model, Mock), "mocking went wrong; backend fixture changed; other tests may fail"

    # Set up the inputs.
    ctx = ChatContext().add(Message("user", "hello"))
    act = CBlock("hello")
    raw_act = CBlock("goodb")
    req_intrinsic = Intrinsic("requirement_check", {"requirement": "did nothing"})
    answerability_intrinsic = Intrinsic("answerability")

    def call_backend_generate():
        """Helper function for generating outputs."""
        return [
            b.generate_from_context(act, ctx),
            b.generate_from_context(req_intrinsic, ctx),
            b.generate_from_context(answerability_intrinsic, ctx),
            b.generate_from_raw([raw_act], ctx, model_options={ModelOption.MAX_NEW_TOKENS: 3})
        ]

    # Call once to populate the memoized mock.
    outputs = await asyncio.gather(*call_backend_generate())
    for output in outputs:
        mot = output[0]
        await mot.avalue() # Ensure all values are computed.

    # Use the memoized mock that errors if not precomputed.
    model.generate = Mock(side_effect=mock_func)
    count = 100 # Use a high number to try to put pressure on the lock and catch deadlocks.
    coros: list[Coroutine[Any, Any, tuple[ModelOutputThunk, Context]]] = []
    for _ in range(count):
        coros.extend(call_backend_generate())

    # Ensure no ordering effects are happening.
    random.shuffle(coros)

    outputs = await asyncio.gather(*coros)
    for output in outputs:
        mot = output[0]
        await mot.avalue()  # Ensure all values get computed.

    faulthandler.disable()


@pytest.mark.qualitative
async def test_generate_with_lock_does_not_block_when_awaiting_value(backend):
    """This is a tricky test to setup. 
    
    It's purpose is to ensure that a long-running generation doesn't get blocked
    when awaiting the `model_output_thunk.avalue()` of a different generation request.

    This means that it is somewhat timing dependent. The generation has to take long enough
    to not instantly resolve but not longer than the timeout. Modify the parameters below to
    finetune this.

    If generation is taking too long, you could just increase the timeout, but that
    causes the test to take longer to run. The best scenario is that the generation doesn't
    resolve before awaiting the other `mot.avalue()` but resolves immediately after.
    """
    # Params to modify depending on speed.
    token_generation_length = 100
    timeout_in_seconds = 30

    # Set up the inputs.
    ctx = ChatContext().add(Message("user", "hello"))
    act = CBlock("hello")
    req_intrinsic = Intrinsic("requirement_check", {"requirement": "did nothing"})
    answerability_intrinsic = Intrinsic("answerability")

    # Create a few model output thunks:
    # - a streaming generation that will take a long time to resolve.
    # - a regular generation that should be able to happen while the streaming is happening.
    # - two intrinsics that shouldn't be able to happen concurrently.
    reg_mot_stream, _ = await backend.generate_from_context(act, ctx, model_options={ModelOption.STREAM: True, ModelOption.MAX_NEW_TOKENS: token_generation_length, "min_length": token_generation_length})
    reg_mot, _ = await backend.generate_from_context(act, ctx)
    req_mot, _ = await backend.generate_from_context(req_intrinsic, ctx, model_options={ModelOption.STREAM: True})
    answerability_mot, _ = await backend.generate_from_context(answerability_intrinsic, ctx, model_options={ModelOption.STREAM: True})

    # Ensure the stream is generating but not yet completing.
    await reg_mot_stream.astream()
    assert not reg_mot_stream.is_computed(), "generation completed too early, see test for more details"

    # Awaiting this shouldn't cause a deadlock. Add the timeout so the test can fail.
    # If the test fails, this means that the streaming generation wasn't able to complete,
    # most likely due to a deadlock caused by awaiting a generation that cannot complete until
    # the streaming is done.
    try:
        async with asyncio.timeout(timeout_in_seconds):
            await req_mot.avalue()
    except Exception as e:
        # The timeout could also be caused by the generation taking too long... be careful!
        # We assume that if the streaming model output thunk is computed after getting its astream here,
        # that it was a deadlock and not the generation taking too long (since the generation is now done).
        await reg_mot_stream.astream()
        if reg_mot_stream.is_computed():
            raise e
        else:
            raise Exception("timeout ended too early, see test for more details")
    
    for output in [reg_mot_stream, reg_mot, req_mot, answerability_mot]:
        if not output.is_computed():
            await output.avalue()  # Ensure everything gets computed.

@pytest.mark.qualitative
async def test_error_during_generate_with_lock(backend):
    # Create local versions of these objects so that mocking
    # doesn't impact other functions. Don't do this in regular code,
    # the copying is complex.
    b: LocalHFBackend = copy(backend)
    model = copy(b._model)
    b._model = model
    b._model.set_adapter([])
    b._added_adapters = {}
    b._loaded_adapters = {}
    b._generate_lock = HFGenerationLock(b)
    b.add_adapter(
        GraniteCommonAdapter(
            "requirement_check", base_model_name=b.base_model_name
        )
    )

    regular_generate = b._model.generate
    def generate_and_raise_exc(*args, **kwargs):
        """Will generate like usual for the intrinsic request. Will fail for the regular generation request."""
        if "max_new_tokens" in kwargs:
            return regular_generate(*args, **kwargs)  # type: ignore
        raise Exception("Oops!")

    b._model.generate = Mock(side_effect=generate_and_raise_exc)
    assert not isinstance(backend._model, Mock), "mocking went wrong; backend fixture changed; other tests may fail"

    # Set up the inputs.
    ctx = ChatContext().add(Message("user", "hello"))
    act = CBlock("hello")
    req_intrinsic = Intrinsic("requirement_check", {"requirement": "did nothing"})

    reg_mot, _ = await b.generate_from_context(act, ctx)
    req_mot, _ = await b.generate_from_context(req_intrinsic, ctx)

    with pytest.raises(Exception, match="Oops!"):
        await reg_mot.avalue()

    await req_mot.avalue()


async def test_generation_lock():
    b = Mock(spec=LocalHFBackend)
    b.load_adapter = Mock()
    b._model = Mock()
    b._model.set_adapter = Mock()
    t = HFGenerationLock(b)

    assert t.backend is b

    # Typically don't use `as` syntax, but useful for asserting things here.
    state = ""
    with t.get_lock(state) as l:
        assert l.state == state
        assert l.lock is t

        assert t.current_state == state
        assert t.num_active == 1

    new_state = "new"
    t.acquire(new_state)
    assert t.current_state  == new_state
    assert t.num_active == 1
    t.release()
    assert t.current_state == new_state, "state only changes when re-acquiring the lock"
    assert t.num_active == 0

    assert str(t) == f"{new_state}: 0"

def test_assert_correct_adapters():
    model = Mock()
    
    # Test scenarios with no active adapters.
    model.active_adapters = Mock(return_value=[])
    _assert_correct_adapters("", model)
    with pytest.raises(AssertionError):
        _assert_correct_adapters("new", model)

    # Test scenarios with one active adapter.
    model.active_adapters = Mock(return_value=["new"])
    with pytest.raises(AssertionError):
        _assert_correct_adapters("", model)
    with pytest.raises(AssertionError):
        _assert_correct_adapters("diff", model)
    _assert_correct_adapters("new", model)

    # Test scenarios when no adapters have been loaded.
    model.active_adapters = Mock(side_effect=ValueError)
    _assert_correct_adapters("", model)
    with pytest.raises(AssertionError):
        _assert_correct_adapters("new", model)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
