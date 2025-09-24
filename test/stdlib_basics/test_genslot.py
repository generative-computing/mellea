import pytest
from typing import Literal
from mellea import generative, start_session
from mellea.stdlib.base import LegacyLinearContext


@generative
def classify_sentiment(text: str) -> Literal["positive", "negative"]: ...


@generative
def write_me_an_email() -> str: ...


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

if __name__ == "__main__":
    pytest.main([__file__])
