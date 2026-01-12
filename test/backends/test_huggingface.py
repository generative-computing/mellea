import asyncio
from copy import copy
import faulthandler
import random
import time
from typing import Any, Coroutine
from unittest.mock import Mock

import pydantic
import pytest
import torch
from typing_extensions import Annotated

from mellea import MelleaSession
from mellea.backends.adapters.adapter import GraniteCommonAdapter
from mellea.backends.cache import SimpleLRUCache
from mellea.backends.formatter import TemplateFormatter
from mellea.backends.huggingface import LocalHFBackend, _assert_correct_adapters
from mellea.backends.types import ModelOption
from mellea.stdlib.base import (
    CBlock,
    ChatContext,
    Context,
    ModelOutputThunk,
    SimpleContext,
)
from mellea.stdlib.chat import Message
from mellea.stdlib.intrinsic.intrinsic import Intrinsic
from mellea.stdlib.requirement import (
    ALoraRequirement,
    LLMaJRequirement,
    Requirement,
    ValidationResult,
    default_output_to_bool,
)


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
        GraniteCommonAdapter("answerability", base_model_name=backend.base_model_name)
    )
    return backend


@pytest.fixture(scope="function")
def session(backend):
    """Fresh HuggingFace session for each test."""
    session = MelleaSession(backend, ctx=ChatContext())
    yield session
    session.reset()


@pytest.mark.qualitative
def test_adapters(backend):
    assert len(backend._added_adapters.items()) > 0

    expected_qualified_name = "requirement_check_alora"
    adapter = backend._added_adapters[expected_qualified_name]
    backend.load_adapter(adapter.qualified_name)
    assert adapter.qualified_name in backend._loaded_adapters

    # Ensure you can load the same adapter twice.
    backend.load_adapter(adapter.qualified_name)

    # Ensure you can unload an adapter.
    backend.unload_adapter(adapter.qualified_name)
    backend.unload_adapter(adapter.qualified_name)
    assert adapter.qualified_name not in backend._loaded_adapters


@pytest.mark.qualitative
def test_system_prompt(session):
    result = session.chat(
        "Where are we going?",
        model_options={ModelOption.SYSTEM_PROMPT: "Talk like a pirate."},
    )
    print(result)


@pytest.mark.qualitative
def test_constraint_lora_with_requirement(session, backend):
    answer = session.instruct(
        "Corporate wants you to find the difference between these two strings: aaaaaaaaaa aaaaabaaaa"
    )
    assert session.backend._cache is not None  # type: ignore
    assert session.backend._use_caches
    assert backend._cache.current_size() != 0
    validation_outputs = session.validate(
        "The answer should mention that there is a b in the middle of one of the strings but not the other."
    )
    assert len(validation_outputs) == 1
    val_result = validation_outputs[0]
    assert isinstance(val_result, ValidationResult)
    assert "requirement_likelihood" in str(val_result.reason)


@pytest.mark.qualitative
def test_constraint_lora_override(session, backend):
    backend.default_to_constraint_checking_alora = False  # type: ignore
    answer = session.instruct(
        "Corporate wants you to find the difference between these two strings: aaaaaaaaaa aaaaabaaaa"
    )
    validation_outputs = session.validate(
        "The answer should mention that there is a b in the middle of one of the strings but not the other."
    )
    assert len(validation_outputs) == 1
    val_result = validation_outputs[0]
    assert isinstance(val_result, ValidationResult)
    assert isinstance(default_output_to_bool(str(val_result.reason)), bool)
    backend.default_to_constraint_checking_alora = True


@pytest.mark.qualitative
def test_constraint_lora_override_does_not_override_alora(session, backend):
    backend.default_to_constraint_checking_alora = False  # type: ignore
    answer = session.instruct(
        "Corporate wants you to find the difference between these two strings: aaaaaaaaaa aaaaabaaaa"
    )
    validation_outputs = session.validate(
        ALoraRequirement(
            "The answer should mention that there is a b in the middle of one of the strings but not the other."
        )
    )
    assert len(validation_outputs) == 1
    val_result = validation_outputs[0]
    assert isinstance(val_result, ValidationResult)
    assert "requirement_likelihood" in str(val_result.reason)

    # Ensure the ValidationResult has its thunk and context set. Ensure the context has
    # the correct actions / results in it.
    assert isinstance(val_result.context, Context)
    assert isinstance(val_result.thunk, ModelOutputThunk)
    assert isinstance(val_result.context.previous_node.node_data, ALoraRequirement)
    assert val_result.context.node_data is val_result.thunk

    backend.default_to_constraint_checking_alora = True


@pytest.mark.qualitative
def test_llmaj_req_does_not_use_alora(session, backend):
    backend.default_to_constraint_checking_alora = True  # type: ignore
    answer = session.instruct(
        "Corporate wants you to find the difference between these two strings: aaaaaaaaaa aaaaabaaaa"
    )
    validation_outputs = session.validate(
        LLMaJRequirement(
            "The answer should mention that there is a b in the middle of one of the strings but not the other."
        )
    )
    assert len(validation_outputs) == 1
    val_result = validation_outputs[0]
    assert isinstance(val_result, ValidationResult)
    assert str(val_result.reason) not in ["Y", "N"]
    assert "requirement_likelihood" not in str(val_result.reason)


@pytest.mark.qualitative
def test_instruct(session):
    result = session.instruct("Compute 1+1.")
    print(result)


@pytest.mark.qualitative
def test_multiturn(session):
    session.instruct("Compute 1+1")
    beta = session.instruct(
        "Take the result of the previous sum and find the corresponding letter in the greek alphabet.",
        model_options={ModelOption.MAX_NEW_TOKENS: 300},
    )
    words = session.instruct("Now list five English words that start with that letter.")
    print(words)


@pytest.mark.qualitative
def test_chat(session):
    output_message = session.chat("What is 1+1?")
    assert "2" in output_message.content, (
        f"Expected a message with content containing 2 but found {output_message}"
    )


@pytest.mark.qualitative
def test_format(session):
    class Person(pydantic.BaseModel):
        name: str
        email_address: Annotated[
            str, pydantic.StringConstraints(pattern=r"[a-zA-Z]{5,10}@example\.com")
        ]

    class Email(pydantic.BaseModel):
        to: Person
        subject: str
        body: str

    output = session.instruct(
        "Write a short email to Olivia, thanking her for organizing a sailing "
        "activity. "
        "Her email is olivia@example.com. "
        "No more than two sentences. ",
        format=Email,
        model_options={ModelOption.MAX_NEW_TOKENS: 2**8},
    )
    print("Formatted output:")
    email = Email.model_validate_json(
        output.value
    )  # this should succeed because the output should be JSON because we passed in a format= argument...
    print(email)

    print("address:", email.to.email_address)
    assert "@" in email.to.email_address, "The @ sign should be in the email address."
    assert email.to.email_address.endswith("example.com"), (
        "The email address should be at example.com"
    )


@pytest.mark.qualitative
async def test_generate_from_raw(session):
    prompts = [
        "what is 1+1?",
        "what is 2+2?",
        "what is 3+3?",
        "what is 4+4?",
        "what is 4+2+2?",
    ]

    results = await session.backend.generate_from_raw(
        actions=[CBlock(value=prompt) for prompt in prompts], ctx=session.ctx
    )

    assert len(results) == len(prompts)
    assert results[0].value is not None


@pytest.mark.qualitative
async def test_generate_from_raw_with_format(session):
    prompts = ["what is 1+1?", "what is 2+2?", "what is 3+3?", "what is 4+4?"]

    class Answer(pydantic.BaseModel):
        name: str
        value: int

    results = await session.backend.generate_from_raw(
        actions=[CBlock(value=prompt) for prompt in prompts],
        format=Answer,
        ctx=session.ctx,
    )

    assert len(results) == len(prompts)

    random_result = results[0]
    try:
        answer = Answer.model_validate_json(random_result.value)
    except pydantic.ValidationError as e:
        assert False, (
            f"formatting directive failed for {random_result.value}: {e.json()}"
        )


@pytest.mark.qualitative
async def test_async_parallel_requests(session):
    model_opts = {ModelOption.STREAM: True}
    mot1, _ = await session.backend.generate_from_context(
        CBlock("Say Hello."), SimpleContext(), model_options=model_opts
    )
    mot2, _ = await session.backend.generate_from_context(
        CBlock("Say Goodbye!"), SimpleContext(), model_options=model_opts
    )

    m1_val = None
    m2_val = None
    if not mot1.is_computed():
        m1_val = await mot1.astream()
    if not mot2.is_computed():
        m2_val = await mot2.astream()

    assert m1_val is not None, "should be a string val after generation"
    assert m2_val is not None, "should be a string val after generation"

    m1_final_val = await mot1.avalue()
    m2_final_val = await mot2.avalue()

    # Ideally, we would be able to assert that m1_final_val != m1_val, but sometimes the first streaming response
    # contains the full response.
    assert m1_final_val.startswith(m1_val), (
        "final val should contain the first streamed chunk"
    )
    assert m2_final_val.startswith(m2_val), (
        "final val should contain the first streamed chunk"
    )

    assert m1_final_val == mot1.value
    assert m2_final_val == mot2.value


@pytest.mark.qualitative
async def test_async_avalue(session):
    mot1, _ = await session.backend.generate_from_context(
        CBlock("Say Hello."), SimpleContext()
    )
    m1_final_val = await mot1.avalue()
    assert m1_final_val is not None
    assert m1_final_val == mot1.value


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
    b.add_adapter(
        GraniteCommonAdapter("requirement_check", base_model_name=b.base_model_name)
    )
    b.add_adapter(
        GraniteCommonAdapter("answerability", base_model_name=b.base_model_name)
    )

    memoized = dict()
    gen_func = model.generate

    def mock_func(input_ids, *args, **kwargs):
        """Mocks the generate function. Must call `populate_mocked_dict` with each input that must be cached before using this."""
        for key, val in memoized.items():
            if torch.equal(key, input_ids):
                time.sleep(random.uniform(0.1, 0.5))  # Simulate a bit of work.
                return val
        assert False, "did not get a cached response"

    # Safely create the dict.
    def populate_mocked_dict(input_ids, *args, **kwargs):
        """Generates the model output and adds to the memoized dict."""
        output = gen_func(input_ids, *args, **kwargs)  # type: ignore
        memoized[input_ids] = output
        return output

    model.generate = Mock(side_effect=populate_mocked_dict)
    assert not isinstance(backend._model, Mock), (
        "mocking went wrong; backend fixture changed; other tests may fail"
    )

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
            b.generate_from_raw(
                [raw_act], ctx, model_options={ModelOption.MAX_NEW_TOKENS: 3}
            ),
        ]

    # Call once to populate the memoized mock.
    outputs = await asyncio.gather(*call_backend_generate())
    for output in outputs:
        mot = output[0]
        await mot.avalue()  # Ensure all values are computed.

    # Use the memoized mock that errors if not precomputed.
    model.generate = Mock(side_effect=mock_func)
    count = (
        5  # Use a high number to try to put pressure on the lock and catch deadlocks.
    )
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
    reg_mot_stream, _ = await backend.generate_from_context(
        act,
        ctx,
        model_options={
            ModelOption.STREAM: True,
            ModelOption.MAX_NEW_TOKENS: token_generation_length,
            "min_length": token_generation_length,
        },
    )
    reg_mot, _ = await backend.generate_from_context(act, ctx)
    req_mot, _ = await backend.generate_from_context(
        req_intrinsic, ctx, model_options={ModelOption.STREAM: True}
    )
    answerability_mot, _ = await backend.generate_from_context(
        answerability_intrinsic, ctx, model_options={ModelOption.STREAM: True}
    )

    # Ensure the stream is generating but not yet completing.
    await reg_mot_stream.astream()
    assert not reg_mot_stream.is_computed(), (
        "generation completed too early, see test for more details"
    )

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
    b.add_adapter(
        GraniteCommonAdapter("requirement_check", base_model_name=b.base_model_name)
    )

    regular_generate = b._model.generate

    def generate_and_raise_exc(*args, **kwargs):
        """Will generate like usual for the intrinsic request. Will fail for the regular generation request."""
        if "max_new_tokens" in kwargs:
            return regular_generate(*args, **kwargs)  # type: ignore
        raise Exception("Oops!")

    b._model.generate = Mock(side_effect=generate_and_raise_exc)
    assert not isinstance(backend._model, Mock), (
        "mocking went wrong; backend fixture changed; other tests may fail"
    )

    # Set up the inputs.
    ctx = ChatContext().add(Message("user", "hello"))
    act = CBlock("hello")
    req_intrinsic = Intrinsic("requirement_check", {"requirement": "did nothing"})

    reg_mot, _ = await b.generate_from_context(act, ctx)
    req_mot, _ = await b.generate_from_context(req_intrinsic, ctx)

    with pytest.raises(Exception, match="Oops!"):
        await reg_mot.avalue()

    await req_mot.avalue()


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
    model.active_adapters = Mock(
        side_effect=ValueError("No adapter loaded. Please load an adapter first.")
    )
    _assert_correct_adapters(
        "", model
    )  # This will fail if peft ever changes the error message.
    with pytest.raises(AssertionError):
        _assert_correct_adapters("new", model)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
