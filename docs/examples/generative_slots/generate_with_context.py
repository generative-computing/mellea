from mellea import generative, start_session
from mellea.backends import ModelOption
from mellea.core import CBlock
from mellea.stdlib.context import ChatContext

# Generative slots can be used with sessions that have context.
# By utilizing context, you can change the results of several
# functions at the same time. This lets you write a function
# once but use it in different ways.


@generative
def grade_essay(essay: str) -> int:
    """Grades the provided essay.

    Args:
        essay: the text to be graded.

    Returns:
        int: a grade between 1 and 100.
    """


@generative
def give_feedback(essay: str) -> list[str]:
    """Generates feedback for improvement for a given essay.

    Args:
        essay: the text that should be commented on.

    Returns:
        list[str]: a list of comments about how the text could be improved.
    """


if __name__ == "__main__":
    m = start_session(
        ctx=ChatContext(), model_options={ModelOption.MAX_NEW_TOKENS: 100}
    )

    text = """
"Mabble" is an obsolete verb meaning to wrap up.
Its only recorded usage is in the early 1600s.
Mabble is a variant of the word "moble", a transitive verb meaning
to muffle (a person, or the head, face, etc...) figuratively
or literally.
"""

    # Generative functions can be used as is, without additional context.
    print("Giving a grade and comments on the essay with no context:")
    grade = grade_essay(m, essay=text)
    print(f"  Grade: {grade}")

    comments = give_feedback(m, essay=text)
    print(f"  Comments: {' '.join(comments)}")

    # If you have a set of generative functions, you can tweak them all by
    # adding context to the session they are running in.
    m.ctx = m.ctx.add(
        CBlock(
            "You are an elementary school teacher. "
            "Any grades and feedback that you give should keep that in mind. Remember to be "
            "especially kind and considerate so that you don't hurt the students' feelings. "
            "Try to keep all grades above an 86 unless something is very wrong."
        )
    )
    print(
        "\n\n\nGiving a grade and comments on the essay with elementary school context:"
    )
    grade = grade_essay(m, essay=text)
    print(f"  Grade: {grade}")

    comments = give_feedback(m, essay=text)
    print(f"  Comments: {' '.join(comments)}")

    # And, let's reset the context and try a different grading style.
    m.reset()
    m.ctx = m.ctx.add(
        CBlock(
            "You are a grammarian that is focused solely on spelling and syntax, "
            "not on the content of essays. When giving grades and feedback, focus "
            "on spelling errors, contractions, and other grammar issues."
        )
    )
    print("\n\n\nGiving a grade and comments on the essay with grammar context:")
    grade = grade_essay(m, essay=text)
    print(f"  Grade: {grade}")

    comments = give_feedback(m, essay=text)
    print(f"  Comments: {' '.join(comments)}")
