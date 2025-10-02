from typing import Literal

from mellea import generative, start_session


@generative
def classify_sentiment(text: str) -> Literal["positive", "negative"]: ...


@generative
def generate_summary(text: str) -> str:
    """This is a function that takes in a string and generates a summary for the string.
    Keep your summary succinct and under 20 words.
    """


if __name__ == "__main__":
    with start_session() as m:
        sentiment_component = classify_sentiment(m, text="I love this!")
        print("Output sentiment is : ", sentiment_component)

        summary = generate_summary(
            text="""
            The eagle rays are a group of cartilaginous fishes in the family Myliobatidae,
            consisting mostly of large species living in the open ocean rather than on the sea bottom.
            Eagle rays feed on mollusks, and crustaceans, crushing their shells with their flattened teeth.
            They are excellent swimmers and are able to breach the water up to several meters above the
            surface. Compared with other rays, they have long tails, and well-defined, rhomboidal bodies.
            They are ovoviviparous, giving birth to up to six young at a time. They range from 0.48 to
            5.1 m (1.6 to 16.7 ft) in length and 7 m (23 ft) in wingspan.
            """
        )
        print("Generated summary is :", summary)
