# pytest: ollama, qualitative, llm, slow

"""Example of chain-of-thought reasoning on a mathematical question from the GSM8K dataset, structured as code for improved performance with Granite 4 models. The original accuracy in standard "thinking" mode is approximately 80%, while this implementation achieves 85-89% accuracy—up to 9 points higher.

This demonstrates that generative decorators are sufficient for complex reasoning tasks: not only do they maintain or improve performance, but they also significantly enhance observability and control. For instance, the structured Thought titles can be easily surfaced in a UI, providing instant insight into the model's reasoning process.
"""

from datasets import load_dataset
from pydantic import BaseModel

from mellea import generative, start_session


class Thought(BaseModel):
    step_name: str
    step_content: str


class ChainOfThought(BaseModel):
    chain_name: str
    step_by_step_solution: list[Thought]


@generative
def compute_chain_of_thought_and_final_answer(question: str) -> ChainOfThought:
    """Generates a comprehensive, explicit chain-of-thought (CoT) solution for the input question,
    with a rigorous focus on correct, cumulative state tracking and operation logic at every step.

    This function decomposes the reasoning process into a sequential list of named, detailed steps,
    each surfacing all logic, calculations, and intermediate values, and explicitly maintaining the
    running state at every stage. The output is a ChainOfThought object containing a descriptive
    chain_name and an ordered list of Thought steps, each with:
      - step_name: A concise, descriptive label for the operation, inference, or transition at that step.
      - step_content: A fully explicit, self-contained explanation of the step's logic, calculation,
        starting and ending state, and the exact operation performed.

    **Principled Rules for Stepwise Reasoning (Best Practices):**
      - **Explicit Running State:**
        After every operation, clearly state:
          - The value(s) *before* the operation.
          - The operation itself (e.g., addition, subtraction).
          - The resulting value(s) *after* the operation.
        Never combine multiple operations into a single step or leave the running total implicit.
      - **Operation Transparency:**
        For every arithmetic or logical operation, write the equation in its canonical form (e.g.,
        "amount before purchase - cost of item = amount after purchase"), and solve for unknowns
        in context. This avoids conflating addition and subtraction, and ensures correct logic.
      - **No Gaps or Ambiguity:**
        Never omit intermediate calculations, even if they seem trivial. Avoid ambiguous references
        or pronouns. Always state exactly what is being operated on, and how the state changes.
      - **Self-Contained Logic:**
        Every step should be understandable in isolation, so the full solution can be reconstructed
        from the steps alone, without prior context.
      - **Descriptive Chain Name:**
        The chain_name should summarize the reasoning process or problem type.

    Args:
        question (str): The question requiring stepwise, explicit reasoning.

    Returns:
        ChainOfThought: An object with a descriptive chain_name and an ordered list of Thought steps,
        each surfacing all operations, values, and logic.

    Examples:
        1. Multi-step Budget Problem
        >>> compute_chain_of_thought_and_final_answer(
        ...     "Alexis starts with $200 and buys a shirt ($30), pants ($46), suit coat ($38), socks ($11), belt ($18), and shoes. After all purchases, she has $16 left. How much did the shoes cost?"
        ... )
        ChainOfThought(
            chain_name="Multi-step Budget Tracking",
            step_by_step_solution=[
                Thought(
                    step_name="Start with budget",
                    step_content="Alexis starts with $200."
                ),
                Thought(
                    step_name="Subtract button-up shirt",
                    step_content="Before purchase: $200. Subtract $30 for the shirt. After purchase: $170."
                ),
                Thought(
                    step_name="Subtract suit pants",
                    step_content="Before purchase: $170. Subtract $46 for the pants. After purchase: $124."
                ),
                Thought(
                    step_name="Subtract suit coat",
                    step_content="Before purchase: $124. Subtract $38 for the suit coat. After purchase: $86."
                ),
                Thought(
                    step_name="Subtract socks",
                    step_content="Before purchase: $86. Subtract $11 for the socks. After purchase: $75."
                ),
                Thought(
                    step_name="Subtract belt",
                    step_content="Before purchase: $75. Subtract $18 for the belt. After purchase: $57."
                ),
                Thought(
                    step_name="Solve for shoes",
                    step_content="Before purchase: $57. Let X be the cost of the shoes. After buying shoes, Alexis has $16. Equation: $57 - X = $16. Solving: X = $41."
                ),
            ]
        )

        2. Logical Riddle
        >>> compute_chain_of_thought_and_final_answer(
        ...     "A farmer has 17 sheep and all but 9 die. How many are left?"
        ... )
        ChainOfThought(
            chain_name="Sheep Survival Logic",
            step_by_step_solution=[
                Thought(
                    step_name="Determine initial number of sheep",
                    step_content="The farmer starts with 17 sheep."
                ),
                Thought(
                    step_name="Interpret 'all but 9 die'",
                    step_content="This phrase means that 9 sheep remain alive; the rest have died."
                ),
                Thought(
                    step_name="State remaining sheep",
                    step_content="Therefore, 9 sheep are left."
                ),
            ]
        )
    """


@generative
def extract_final_short_answer(
    question: str, chain_of_thought: ChainOfThought
) -> int: ...


if __name__ == "__main__":
    scores = []
    m = start_session()

    for question, target in (
        x.values() for x in load_dataset("gsm8k", "main", split="train[:100]")
    ):
        target = int(target.split("####")[-1])
        response = compute_chain_of_thought_and_final_answer(m, question=question)
        for step in response.step_by_step_solution:
            print(step.step_name)
            print(step.step_content)
        answer = extract_final_short_answer(
            m, question=question, chain_of_thought=response
        )
        print("Answer: ", answer)
        print("Target: ", target)
        scores.append(target == answer)
    print(f"Final Score: {((sum(scores) / len(scores)) * 100):.1f}/100")
