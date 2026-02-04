# pytest: ollama, llm

from typing import Literal

from mellea import generative, start_session


@generative
def classify_sentiment(text: str) -> Literal["positive", "negative"]:
    """Classify the sentiment of the input text as 'positive' or 'negative'."""


m = start_session()
sentiment = classify_sentiment(m, text="I love this!")
print("Output sentiment is:", sentiment)
