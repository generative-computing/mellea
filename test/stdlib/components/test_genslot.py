import asyncio
from typing import Literal

import pytest

from mellea import MelleaSession, generative, start_session
from mellea.backends.model_ids import IBM_GRANITE_4_MICRO_3B
from mellea.backends.ollama import OllamaModelBackend
from mellea.core import Requirement
from mellea.stdlib.components.genslot import (
    AsyncGenerativeSlot,
    GenerativeSlot,
    PreconditionException,
    SyncGenerativeSlot,
)
from mellea.stdlib.context import ChatContext, Context
from mellea.stdlib.requirements import simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy

# Module-level markers: Uses granite3.3:8b (8B, heavy) in local mode
pytestmark = [
    pytest.mark.ollama,
    pytest.mark.requires_gpu,
    pytest.mark.requires_heavy_ram,
    pytest.mark.llm,
]


@pytest.fixture(scope="module")
def backend(gh_run: int):
    """Shared backend."""
    if gh_run == 1:
        return OllamaModelBackend(
            model_id=IBM_GRANITE_4_MICRO_3B.ollama_name  # type: ignore
        )
    else:
        return OllamaModelBackend(model_id="granite3.3:8b")


@generative
def classify_sentiment(text: str) -> Literal["positive", "negative"]: ...


@generative
def write_me_an_email() -> str: ...


@generative
async def async_write_short_sentence(topic: str) -> str: ...


@pytest.fixture(scope="function")
def session():
    """Fresh session for each test."""
    session = start_session()
    yield session
    session.reset()


@pytest.fixture
def classify_sentiment_output(session):
    return classify_sentiment(session, text="I love this!")


def test_gen_slot_output(classify_sentiment_output):
    assert isinstance(classify_sentiment_output, str)


def test_func(session):
    assert isinstance(write_me_an_email, SyncGenerativeSlot)
    write_email_component = write_me_an_email(session)
    assert isinstance(write_email_component, str)


@pytest.mark.qualitative
def test_sentiment_output(classify_sentiment_output):
    assert classify_sentiment_output in ["positive", "negative"]


def test_gen_slot_logs(classify_sentiment_output, session):
    sent = classify_sentiment_output
    last_prompt = session.last_prompt()[-1]
    assert isinstance(last_prompt, dict)
    assert set(last_prompt.keys()) == {"role", "content", "images"}


def test_gen_slot_with_context_and_backend(session):
    email, context = write_me_an_email(context=session.ctx, backend=session.backend)
    assert isinstance(email, str)
    assert isinstance(context, Context)


async def test_async_gen_slot(session):
    assert isinstance(async_write_short_sentence, AsyncGenerativeSlot)

    r1 = async_write_short_sentence(session, topic="cats")
    r2 = async_write_short_sentence(session, topic="dogs")

    r3, c3 = await async_write_short_sentence(
        context=session.ctx, backend=session.backend, topic="fish"
    )
    results = await asyncio.gather(r1, r2)

    assert isinstance(r3, str)
    assert isinstance(c3, Context)
    assert len(results) == 2


@pytest.mark.parametrize(
    "arg_choices,kwarg_choices,errs",
    [
        pytest.param(["m"], ["func1", "func2", "func3"], False, id="session"),
        pytest.param(["context"], ["backend"], False, id="context and backend"),
        pytest.param(
            ["backend"], ["func1", "func2", "func3"], True, id="backend without context"
        ),
        pytest.param(["m"], ["m"], True, id="duplicate arg and kwarg"),
        pytest.param(
            [
                "m",
                "precondition_requirements",
                "requirements",
                "strategy",
                "model_options",
                "func1",
                "func2",
                "func3",
            ],
            [],
            True,
            id="original func args as positional args",
        ),
        pytest.param(
            [], ["m", "func1", "func2", "func3"], False, id="session and func as kwargs"
        ),
        pytest.param(
            [],
            [
                "m",
                "precondition_requirements",
                "requirements",
                "strategy",
                "model_options",
                "func1",
                "func2",
                "func3",
            ],
            False,
            id="all kwargs",
        ),
        pytest.param(
            [],
            ["func1", "m", "func2", "requirements", "func3"],
            False,
            id="interspersed kwargs",
        ),
        pytest.param([], [], True, id="missing required args"),
    ],
)
def test_arg_extraction(backend, arg_choices, kwarg_choices, errs):
    """Tests the internal extract_args_and_kwargs function.

    This function has to test a large number of input combinations; as a result,
    it uses a parameterization scheme. It takes a list and a dict. Each contains
    strings corresponding to the possible args/kwargs below. Order matters in the list.
    See the param id for an idea of what the test does.

    Python should catch most of these issues itself. We have to manually raise an exception for
    the arguments of the original function being positional.
    """
    # List of all needed values.
    backend = backend
    ctx = ChatContext()
    session = MelleaSession(backend, ctx)
    precondition_requirements = ["precondition"]
    requirements = None
    strategy = RejectionSamplingStrategy()
    model_options = {"test": 1}
    func1 = 1
    func2 = True
    func3 = "func3"

    # Lookup table by name.
    vals = {
        "m": session,
        "backend": backend,
        "context": ctx,
        "precondition_requirements": precondition_requirements,
        "requirements": requirements,
        "strategy": strategy,
        "model_options": model_options,
        "func1": func1,
        "func2": func2,
        "func3": func3,
    }

    args = []
    for arg in arg_choices:
        args.append(vals[arg])

    kwargs = {}
    for kwarg in kwarg_choices:
        kwargs[kwarg] = vals[kwarg]

    # Run the extraction and check for the (un-)expected exception.
    found_err = False
    err = None
    try:
        GenerativeSlot.extract_args_and_kwargs(*args, **kwargs)
    except Exception as e:
        found_err = True
        err = e

    if errs:
        assert found_err, "expected an exception and got none"
    else:
        assert not found_err, f"got unexpected err: {err}"


def test_disallowed_parameter_names():
    with pytest.raises(ValueError):

        @generative
        def test(backend): ...


def test_precondition_failure(session):
    with pytest.raises(PreconditionException):
        classify_sentiment(
            m=session,
            text="hello",
            precondition_requirements=[
                Requirement(
                    "forced failure",
                    validation_fn=simple_validate(lambda x: (False, "")),
                )
            ],
        )


def test_requirement(session):
    classify_sentiment(
        m=session, text="hello", requirements=["req1", "req2", Requirement("req3")]
    )


def test_with_no_args(session):
    @generative
    def generate_text() -> str:
        """Generate text!"""

    generate_text(m=session)


if __name__ == "__main__":
    pytest.main([__file__])
