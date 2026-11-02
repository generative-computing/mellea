# pytest: ollama, e2e
"""Building a sentiment classification system.

Prerequisites:
    ollama serve
    ollama pull granite4.2:3b

Usage:
    uv run python docs/examples/tutorial/sentiment_classifier.py
"""

from typing import Literal

from mellea import generative, start_session


@generative
def classify_sentiment(text: str) -> Literal["positive", "negative"]:
    """Classify the sentiment of the input text as 'positive' or 'negative'."""


m = start_session()
sentiment = classify_sentiment(m, text="I love this!")
print("Output sentiment is:", sentiment)
