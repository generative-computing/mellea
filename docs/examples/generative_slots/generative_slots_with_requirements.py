from typing import Literal

from mellea import generative, start_session
from mellea.core import Requirement
from mellea.stdlib.components.genslot import PreconditionException
from mellea.stdlib.requirements import simple_validate
from mellea.stdlib.sampling.base import RejectionSamplingStrategy


@generative
def classify_sentiment(text: str) -> Literal["positive", "negative", "unknown"]:
    """Classify the sentiment of the text."""


if __name__ == "__main__":
    m = start_session()

    # Add preconditions and requirements.
    sentiment_component = classify_sentiment(
        m,
        text="I love this!",
        # Preconditions are only checked with basic validation. Don't use the strategy.
        precondition_requirements=["the text arg should be less than 100 words"],
        # Reqs to use with the strategy. You could also just remove "unknown" from the structured output for this.
        requirements=["avoid classifying the sentiment as unknown"],
        strategy=RejectionSamplingStrategy(),  # Must specify a strategy for gen slots
    )

    print(
        f"Prompt to the model looked like:\n```\n{m.last_prompt()[0]['content']}\n```"  # type: ignore[index]
    )
    # Prompt to the model looked like:
    # ```
    # Your task is to imitate the output of the following function for the given arguments.
    # Reply Nothing else but the output of the function.

    # Function:
    # def classify_sentiment(text: str) -> Literal['positive', 'negative', 'unknown']:
    #     """Classify the sentiment of the text.

    #     Postconditions:
    #         - avoid classifying the sentiment as unknown
    #     """

    # Arguments:
    # - text: "I love this!"  (type: <class 'str'>)
    # ```

    print("\nOutput sentiment is:", sentiment_component)

    # We can also force a precondition failure.
    try:
        sentiment_component = classify_sentiment(
            m,
            text="I hate this!",
            # Requirement always fails to validate given the lambda.
            precondition_requirements=[
                Requirement(
                    "the text arg should be only one word",
                    validation_fn=simple_validate(lambda x: (False, "Forced to fail!")),
                )
            ],
        )
    except PreconditionException as e:
        print(f"exception: {e!s}")

        # Look at why the precondition validation failed.
        print("Failure reasons:")
        for val_result in e.validation:
            print("-", val_result.reason)
