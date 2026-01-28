"""
Example usage of budget forcing in long-chain-of-thought reasoning tasks.

To run this script from the root of the Mellea source tree, use the command:
```
uv run python docs/examples/budget_forcing/budget_forcing_example.py
```
"""

from mellea import MelleaSession, start_session
from mellea.backends import ModelOption
from mellea.backends.model_ids import IBM_GRANITE_4_MICRO_3B
from mellea.stdlib.sampling.budget_forcing import BudgetForcingSamplingStrategy


def solve_on_budget(
    m_session: MelleaSession, prompt: str, thinking_budget: int = 512
) -> str:
    """Solves the problem in `prompt`, force-stopping thinking at `thinking_budget` tokens
    (if reached), and returns the solution"""
    # Sampling strategy for budget forcing: pass the thinking budget here
    strategy = BudgetForcingSamplingStrategy(
        think_max_tokens=thinking_budget,
        start_think_token="<think>",
        end_think_token="</think>",
        answer_suffix="The final answer is:",
        requirements=None,
    )

    # Perform greedy decoding, not exceeding the thinking token budget
    result = m_session.instruct(
        prompt, strategy=strategy, model_options={ModelOption.TEMPERATURE: 0}
    )
    output_str = str(
        result
    )  # solution containing (a) a thinking section within <think> and </think> (possibly incomplete due to budget forcing), and (b) a final answer

    return output_str


# Create a Mellea session for granite-4.0-micro with an Ollama backend
m_session = start_session(backend_name="ollama", model_id=IBM_GRANITE_4_MICRO_3B)

# Demonstrate granite solving the same problem on various thinking budgets
prompt = "To double your investment in 5 years, what must your annual return be? Put your final answer within \\boxed{}."
different_thinking_budgets = [256, 64, 16]  # max number of thinking tokens allowed
for thinking_budget in different_thinking_budgets:
    solution = solve_on_budget(m_session, prompt, thinking_budget=thinking_budget)
    header = f"MAX THINKING BUDGET: {thinking_budget} tokens"
    print(f"{'-' * len(header)}\n{header}\n{'-' * len(header)}")
    print(f"PROMPT: {prompt}")
    print(f"\nSOLUTION: {solution}")
    print(f"\n\nSOLUTION LENGTH: {len(solution)} characters")
    print(f"{'-' * len(header)}\n\n")
