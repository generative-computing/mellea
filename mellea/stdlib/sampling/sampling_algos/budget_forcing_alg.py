"""Budget-forcing generation algorithm for thinking models.

Implements `think_budget_forcing`, which extends a model's reasoning phase by
repeatedly appending a "think more" suffix whenever the model attempts to close its
`<think>` block prematurely, following the method proposed in arXiv:2501.19393.
Generation is split into a thinking pass (bounded by `think_max_tokens`) and an
answer pass (bounded by `answer_max_tokens`), using the raw completions API of an
`OllamaModelBackend`.
"""

from ....backends import ModelOption
from ....backends.ollama import OllamaModelBackend
from ....core import (
    BaseModelSubclass,
    CBlock,
    Component,
    Context,
    GenerateLog,
    ModelOutputThunk,
)


async def think_budget_forcing(
    backend: OllamaModelBackend,
    action: CBlock | Component | ModelOutputThunk,
    *,
    ctx: Context,
    format: type[BaseModelSubclass] | None = None,
    tool_calls: bool = False,
    think_max_tokens: int | None = 4096,
    answer_max_tokens: int | None = None,
    start_think_token: str | None = "<think>",
    end_think_token: str | None = "</think>",
    begin_response_token: str | None = "",
    think_more_suffix: str | None = "",
    answer_suffix: str | None = "",
    model_options: dict | None = None,
) -> ModelOutputThunk:
    r"""Generate with budget forcing using the completions APIs.

    This relies on raw autocompletion and assumes the model's output is structured in the following form: '<think> ... </think> summary answer'
    The budget forcing method is proposed in the paper: https://arxiv.org/abs/2501.19393
    This implementation tries to follow the key outlines in the paper while ensuring stable and fail-safe operation.
    This is performed via multi-step generation. The model will be called multiple times until requirements are met, in other words, the response will be assembled conditionally.

    Args:
        backend: OllamaModelBackend instance to use for generation.
        action: The last item of the context, passed as an `action` instead of as part
            of the `ctx`. See `docs/dev/generate_signature_decisions.md`.
        ctx: The current conversation context.
        format: Optional Pydantic model for constrained decoding of the response.
        tool_calls: If `True`, tool calling is enabled.
        think_max_tokens: Budget in number of tokens allocated for the think block.
        answer_max_tokens: Budget in number of tokens allocated for the summary and
            answer block; `None` indicates unbounded answer, generating till EoS.
        start_think_token: String indicating start of think block, default `<think>`.
        end_think_token: String indicating end of think block, default `</think>`.
        begin_response_token: Used by certain models, string indicating start of
            response block, e.g. `"<response>"`, default `""`.
        think_more_suffix: String to append to force continued thinking, e.g.
            `"\nWait"`; if `None`, additional thinking is not forced (upper-bound
            budget case).
        answer_suffix: String to append to force a final answer.
        model_options: Any model options to upsert into the defaults for this call.

    Returns:
        ModelOutputThunk: The assembled thinking and answer response.

    Raises:
        RuntimeError: If a sub-call's `generation.usage` is `None` (no token
            counts available to accumulate against the budget).

    Assumptions:
        -  The chat template is applied on prompt, with think mode enabled
        -  Model is think mode activated
        -  enabling prefix-caching improves performance

    Limitations:
        -  Does not support batching
    """
    responses = []
    prompt = backend.formatter.print(action)
    if start_think_token:
        prompt += start_think_token
        responses.append(start_think_token)

    # Generate thinking portion
    if model_options is None:
        model_options = dict()
    model_options["n"] = 1
    if think_max_tokens is None:
        think_max_tokens = 0
    rem_toks = think_max_tokens
    model_options[ModelOption.MAX_NEW_TOKENS] = rem_toks
    gen_tok_count = 0
    curr_prompt = prompt
    _generate_logs: list[GenerateLog | None] = []
    _prompt_tokens: int | None = None  # captured from the first sub-call
    min_char_len = 10

    # think block indefinite multi-step operation to satisfy user's budget
    while True:
        if rem_toks <= 0:  # zero-think case
            break

        model_options[ModelOption.MAX_NEW_TOKENS] = rem_toks
        result = await backend.generate_from_raw(
            [CBlock(value=curr_prompt)],
            model_options=model_options,
            ctx=ctx,
            tool_calls=tool_calls,
            format=format,
        )
        _generate_logs.append(result[0]._generate_log)
        usage = result[0].generation.usage
        if usage is None:
            raise RuntimeError(
                "think_budget_forcing requires per-call token counts; "
                "backend returned `mot.generation.usage = None`."
            )
        if _prompt_tokens is None:
            _prompt_tokens = usage["prompt_tokens"]
        gen_tok_count += usage["completion_tokens"]
        rem_toks = think_max_tokens - gen_tok_count
        response = result[0].value if result[0].value else ""

        if think_more_suffix is None or think_more_suffix == "":
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
            if len(step.strip()) <= min_char_len:  # type: ignore
                responses.append(response)
                break

            # request more steps
            step = f"{step} {think_more_suffix}"  # type: ignore
            responses.append(step)
            curr_prompt += step

    response = "".join(responses)

    if answer_suffix is None:
        if _prompt_tokens is None:
            raise RuntimeError(
                "think_budget_forcing produced no generations; "
                "check `think_max_tokens` and `answer_suffix`."
            )
        # create response ModelOutputThunk object
        _res = ModelOutputThunk(value=response)
        _res.generation.usage = {
            "prompt_tokens": _prompt_tokens,
            "completion_tokens": gen_tok_count,
            "total_tokens": _prompt_tokens + gen_tok_count,
        }
        # we will simply take the last log output as a representative log, alternatively we can merge the logs but that function is not available yet
        _res._generate_log = _generate_logs[-1]
        return _res

    #  One more round of generate to get an answer
    if end_think_token and end_think_token not in response:
        response += f" {end_think_token}"

    if begin_response_token and begin_response_token not in response:
        response += f" {begin_response_token}"

    if answer_suffix:
        response += f" {answer_suffix}"

    # update original curr_prompt with assembled response
    curr_prompt += response
    if answer_max_tokens is not None:
        model_options[ModelOption.MAX_NEW_TOKENS] = answer_max_tokens

    else:
        model_options.pop(ModelOption.MAX_NEW_TOKENS, None)  # generate unconditionally

    # model_options["logprobs"] = 1  # To get number of generated tokens
    result = await backend.generate_from_raw(
        [CBlock(curr_prompt)],
        model_options=model_options,
        ctx=ctx,
        tool_calls=tool_calls,
        format=format,
    )
    _generate_logs.append(result[0]._generate_log)
    response += result[0].value if result[0].value else ""
    usage = result[0].generation.usage
    if usage is None:
        raise RuntimeError(
            "think_budget_forcing requires per-call token counts; "
            "backend returned `mot.generation.usage = None`."
        )
    if _prompt_tokens is None:
        # zero-think case: capture prompt count from this answer pass.
        _prompt_tokens = usage["prompt_tokens"]
    gen_tok_count += usage["completion_tokens"]
    # create response ModelOutputThunk object
    _res = ModelOutputThunk(value=response)
    _res.generation.usage = {
        "prompt_tokens": _prompt_tokens,
        "completion_tokens": gen_tok_count,
        "total_tokens": _prompt_tokens + gen_tok_count,
    }
    # we will simply take the last log output as a representative log, alternatively we can merge the logs but that function is not available yet
    _res._generate_log = _generate_logs[-1]
    return _res
