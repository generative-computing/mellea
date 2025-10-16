"""Budget forcing implementation."""

import re

from mellea.backends.ollama import OllamaModelBackend
from mellea.stdlib.base import CBlock, Component, GenerateLog
from mellea.stdlib.session import MelleaSession


def think_budget_forcing(
    session: MelleaSession,
    action: CBlock | Component,
    *,
    think_max_tokens: int = 4096,
    answer_max_tokens: int | None = None,
    start_think_token: str = "<think>",
    end_think_token: str = "</think>",
    begin_response_token: str = "",
    end_response_token: str = "",
    think_wait_suffix: str = "",
    answer_suffix: str = "The final answer is:",
    answer_regex: str = r"\\boxed{.*?}",
    model_options: dict | None = None,
    generate_logs: list[GenerateLog] | None = None,
):
    r"""Generate with budget forcing using the completions APIs.

    This relies on raw autocompletion and assumes the model's output is structured in the following form: '<think> ... </think> summary answer'
    The budget forcing method is proposed in the paper: https://arxiv.org/abs/2501.19393
    This implementation tries to follow the key outlines in the paper while ensuring stable and fail-safe operation.
    This is performed via multi-step generation. The model will be called multiple times until requirements are met, in other words, the response will be assembled conditionally.

    Args:
        session: Mellea session
        action: The last item of the context should be passed in as an `action` instead of as part of the `ctx`. See `docs/dev/generate_signature_decisions.md`.
        think_max_tokens: Budget in number of tokens allocated for the think block
        answer_max_tokens: Budget in number of tokens allocated for the summary and answer block, None indicates generating till EoS
        start_think_token: String indicating start of think block, default <think>
        end_think_token: String indicating end of think block, default </think>
        begin_response_token: Used by certain models, string indicating start of response block, e.g. "<response>", default None
        end_response_token: Used by certain models, string indicating end of response block, e.g. "</response>", default None
        think_wait_suffix: String to append to force continued thinking, e.g. "\nWait" if set to None we will not force additional thinking. Use None for upper-bound budget case
        answer_suffix: String to append to force a final answer
        answer_regex: Answer regex which indicates an answer is generated
        model_options: Any model options to upsert into the defaults for this call.
        generate_logs: a `GenerateLog` instance to add log information to.

    Assumptions:
        -  The chat template is applied on prompt, with think mode enabled
        -  Model is think mode activated
        -  enabling prefix-caching improves performance

    Limitations:
        -  Does not support batching
    """
    backend = session.backend
    # TODO should expand backend support
    assert isinstance(backend, OllamaModelBackend)
    model_options = backend._simplify_and_merge(model_options)

    responses = []
    prompt = backend.formatter.print(action)
    if start_think_token:
        prompt += start_think_token
        responses.append(start_think_token)

    # Generate thinking portion
    # model_options["echo"] = True
    # model_options["logprobs"] = 1
    model_options["n"] = 1
    rem_toks = think_max_tokens
    gen_tok_count = 0
    curr_prompt = prompt
    min_step_len = 10  # minimum character length of step to be considered valid

    # think block indefinite multi-step operation to satisfy user's budget
    while True:
        if rem_toks <= 0:  # zero-think case
            break

        if rem_toks <= min_step_len:  # minimum step length reached
            break

        model_options["max_tokens"] = rem_toks
        # TODO workaround to obtain generated token counts
        # The token count should be relayed by openai's CompletionUsage
        # model_options["logprobs"] = 1  # To get number of generated tokens
        result = backend._generate_from_raw(
            [CBlock(value=curr_prompt)],
            model_options=model_options,
            generate_logs=generate_logs,
        )
        # gen_tok_count += len(result[0]._meta['oai_completion_response']['logprobs']['token_logprobs'])
        gen_tok_count += result[0]._meta["generate_response"]["eval_count"]
        rem_toks = think_max_tokens - gen_tok_count
        response = result[0].value if result[0].value else ""

        if think_wait_suffix == "":
            # non-strict budget form
            responses.append(response)
            break

        if rem_toks <= 0:
            responses.append(response)
            break

        else:
            if end_think_token:
                step = response.split(end_think_token)[0]
            # model fails to produce thoughts, let's exit
            if len(step.strip()) <= min_step_len:
                responses.append(response)
                break

            # request more steps
            step = f"{step} {think_wait_suffix}"
            responses.append(step)
            curr_prompt += step

    response = "".join(responses)
    if answer_regex is None or answer_suffix is None:
        return response, gen_tok_count

    # Now get a final answer if we need to
    # TODO: Here we check if a final answer exists, technically we should check for an answer outside
    # The think block, but we will use relaxed requirement of finding any answer in the model's response.
    # Consider a strict structural approach in the future.
    # e.g.
    # answer_blk = response.split(end_think_token)[-1]

    # Check if answer in response
    matches = re.findall(answer_regex, response, re.DOTALL)
    if len(matches) > 0:
        return response, gen_tok_count

    # Answer is not in response, let's force an answer
    if end_think_token and end_think_token not in response:
        response += f" {end_think_token}"

    if begin_response_token and begin_response_token not in response:
        response += f" {begin_response_token}"

    if answer_suffix:
        response += f" {answer_suffix}"

    # update original curr_prompt with assembled response
    curr_prompt += response
    if answer_max_tokens is not None:
        model_options["max_tokens"] = answer_max_tokens

    else:
        model_options.pop("max_tokens", None)  # generate unconditionally

    # model_options["logprobs"] = 1  # To get number of generated tokens
    result = backend._generate_from_raw(
        [CBlock(curr_prompt)], model_options=model_options, generate_logs=generate_logs
    )
    response += result[0].value if result[0].value else ""
    gen_tok_count += result[0]._meta["generate_response"]["eval_count"]
    # gen_tok_count += len(result[0]._meta['oai_completion_response']['logprobs']['token_logprobs'])
    return response, gen_tok_count
