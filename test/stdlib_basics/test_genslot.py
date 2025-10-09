import asyncio
import pytest
from typing import Literal
from mellea import generative, start_session
from mellea.stdlib.base import Context
from mellea.stdlib.genslot import AsyncGenerativeSlot, GenerativeSlot, SyncGenerativeSlot


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

    r3, c3 = await async_write_short_sentence(context=session.ctx, backend=session.backend, topic="fish")
    results = await asyncio.gather(r1, r2)

    assert isinstance(r3, str)
    assert isinstance(c3, Context)
    assert len(results) == 2

def test_duplicate_args(session):
    with pytest.raises(ValueError, match="passed in multiple values"):
        _ = write_me_an_email(session.ctx, backend=session.backend, context=session.ctx)  # type: ignore

def test_extra_args(session):
    with pytest.raises(TypeError, match="got an unexpected keyword argument"):
        _ = write_me_an_email(m=session, random_param="random_param")  # type: ignore

def test_without_required_args():
    with pytest.raises(ValueError, match="need to pass in a session or a"):
        _ = write_me_an_email()  # type: ignore


if __name__ == "__main__":
    pytest.main([__file__])
