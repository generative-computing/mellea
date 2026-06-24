# SPDX-License-Identifier: Apache-2.0

"""Common utility functions for the library and tests."""

# Standard
from __future__ import annotations

import contextlib
import itertools
import json
import logging
import os
import uuid
from typing import TYPE_CHECKING

# Third Party
import pydantic

from ....core.utils import MelleaLogger

if TYPE_CHECKING:
    import llguidance
    import torch
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

# First Party
from .types import ChatCompletionResponse, ChatCompletionResponseChoice


@contextlib.contextmanager
def import_optional(extra_name: str):
    """Handle optional imports.

    Args:
        extra_name: Package extra to suggest in the install hint
            (e.g. `pip install mellea[extra_name]`).
    """
    try:
        yield
    except ImportError as err:
        logging.warning(
            "%s.\nHINT: You may need to pip install %s[%s]",
            err,
            __package__,
            extra_name,
        )
        raise


def random_uuid() -> str:
    """Generate a random UUID string.

    Returns:
        Hexadecimal UUID string suitable for use as a unique identifier.
    """
    return str(uuid.uuid4())


def load_transformers_lora(local_or_remote_path: str) -> tuple:
    """Load transformers LoRA model placed on the best available device.

    AutoModelForCausalLM.from_pretrained() is supposed to auto-load base models if you
    pass it a LoRA adapter's config, but that auto-loading is very broken as of 8/2025.
    Workaround powers activate!

    Device selection mirrors ``LocalHFBackend``: CUDA → MPS → CPU. The returned model
    is already on that device; callers do not need to move it.

    Only works if `transformers` and `peft` are installed.

    Args:
        local_or_remote_path: Local directory path of the LoRA adapter.

    Returns:
        Tuple of `(model, tokenizer)` where `model` is the loaded LoRA model placed on
        the best available device, and `tokenizer` is the corresponding HuggingFace
        tokenizer.

    Raises:
        ImportError: If `peft` or `transformers` packages are not installed.
        NotImplementedError: If `local_or_remote_path` does not exist locally
            (remote loading from the Hugging Face Hub is not yet implemented).
    """
    with import_optional("hf"):
        # Third Party
        import torch
    with import_optional("peft"):
        # Third Party
        import peft
        import transformers
    local_model_dir = local_or_remote_path
    if not os.path.exists(local_model_dir):
        raise NotImplementedError("TODO: Talk to hugging face hub")
    with open(f"{local_model_dir}/adapter_config.json", encoding="utf-8") as f:
        adapter_config = json.load(f)
    base_model_name = adapter_config["base_model_name_or_path"]
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    base_model = transformers.AutoModelForCausalLM.from_pretrained(
        base_model_name, device_map=str(device)
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(base_model_name)
    model = peft.PeftModel.from_pretrained(
        base_model, local_model_dir, device_map=str(device)
    )
    return model, tokenizer


class _GuidanceLogitsProcessor:
    """A HuggingFace logits processor that enforces an llguidance grammar."""

    # Modified from VLLM v0.9.2 code base
    # https://github.com/vllm-project/vllm/blob/v0.9.2/vllm/model_executor/guided_decoding/guidance_logits_processors.py

    def __init__(self, grammar: str, ll_tokenizer: llguidance.LLTokenizer) -> None:
        """Initialize the processor with a compiled grammar and an llguidance tokenizer."""
        with import_optional("llguidance"):
            # Callers will have already had to import llguidance. Ensure it here.
            import llguidance
            import llguidance.torch

        self.grammar = grammar
        self.vocab_size: int = ll_tokenizer.vocab_size
        self.ll_tokenizer: llguidance.LLTokenizer = ll_tokenizer
        self.ll_matchers: list[llguidance.LLMatcher] = []
        self.bitmasks: list[torch.Tensor] = []
        self.new_sampling: bool = False
        self.batch_size: int = -1

    def __call__(
        self, batch_input_ids: torch.Tensor, batch_scores: torch.Tensor
    ) -> torch.Tensor:
        """Apply the grammar's allowed-token bitmask to ``batch_scores`` in place."""
        # Guaranteed to be imported by class __init__.
        import llguidance
        import llguidance.torch

        i_batch, _ = batch_input_ids.shape
        s_batch, _ = batch_scores.shape
        if i_batch != s_batch:
            raise RuntimeError(
                f"batch size mismatch: input_ids={i_batch}, scores={s_batch}"
            )

        if self.batch_size != i_batch:
            self.batch_size = i_batch
            self.bitmasks = [
                llguidance.torch.allocate_token_bitmask(1, self.vocab_size)  # type: ignore[attr-defined]
                for _ in range(self.batch_size)
            ]
            self.ll_matchers = [
                llguidance.LLMatcher(self.ll_tokenizer, self.grammar)
                for _ in range(self.batch_size)
            ]

        for input_ids, scores, ll_matcher, bitmask in zip(
            batch_input_ids, batch_scores, self.ll_matchers, self.bitmasks
        ):
            if self.new_sampling and len(input_ids) > 0:
                ll_matcher.consume_token(input_ids.tolist()[-1])  # type: ignore[attr-defined]
                err = ll_matcher.get_error()  # type: ignore[attr-defined]
                if err:
                    MelleaLogger.get_logger().warning("Error in LLMatcher: %s", err)

            llguidance.torch.fill_next_token_bitmask(ll_matcher, bitmask, 0)
            llguidance.torch.apply_token_bitmask_inplace(  # type: ignore[attr-defined]
                scores, bitmask.to(scores.device)
            )

        self.new_sampling = True
        return batch_scores


def chat_completion_request_to_transformers_inputs(
    request: dict,
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    constrained_decoding_prefix: str | None = None,
    ll_tokenizer: llguidance.LLTokenizer | None = None,
) -> tuple[dict, dict]:
    """Translate an OpenAI-style chat completion request.

    Translate an OpenAI-style chat completion request into an input for a Transformers
    `generate()` call.

    Args:
        request: Request as parsed JSON or equivalent dataclass.
        tokenizer: HuggingFace tokenizer.
        model: HuggingFace model object. Used for `model.device` placement and
            when `constrained_decoding_prefix` is set.
        constrained_decoding_prefix: Optional generation prefix to append to the prompt.
        ll_tokenizer: Pre-built `llguidance.LLTokenizer`. Only used when the request
            uses constrained decoding; if not provided, one is constructed from
            `tokenizer`. Pass an existing instance to avoid the construction cost.

    Returns:
        Tuple of `(generate_input, other_input)` where `generate_input` contains
        kwargs to pass directly to `generate()` and `other_input` contains
        additional parameters for `generate_with_transformers`.

    Raises:
       ImportError: If `torch`, `transformers`, or `llguidance` packages
            are not installed (the latter only when constrained decoding is used).
        TypeError: If `tokenizer.apply_chat_template()` returns an unexpected type.
        ValueError: If padding or end-of-sequence token IDs cannot be determined
            from the tokenizer.
    """
    with import_optional("torch"):
        # Third Party
        import torch
    with import_optional("transformers"):
        # Third Party
        import transformers

    if isinstance(request, pydantic.BaseModel):
        request = request.model_dump()

    generate_input: dict = {
        # Always return dict, else downstream code will need lots type checks
        "return_dict_in_generate": True
    }

    tokenizer_input = {
        "conversation": request["messages"],
        "add_generation_prompt": True,
    }

    if request.get("tools") is not None:
        tokenizer_input["tools"] = request["tools"]

    # pylint: disable=unsupported-membership-test
    if (
        request.get("extra_body") is not None
        and request["extra_body"].get("documents") is not None
    ):
        tokenizer_input["documents"] = request["extra_body"]["documents"]

    input_tokens = tokenizer.apply_chat_template(**tokenizer_input, return_tensors="pt")  # type: ignore[union-attr]

    # Transformers 5 switched the return type of apply_chat_template() from Tensor to
    # BatchEncoding. Adjust our behavior depending on which direction the currently
    # installed version of apply_chat_template() decided to go.
    if isinstance(input_tokens, transformers.tokenization_utils_base.BatchEncoding):
        # BatchEncoding
        input_tokens = input_tokens["input_ids"]
    elif not isinstance(input_tokens, torch.Tensor):
        raise TypeError(
            f"Expected Tokenizer.apply_chat_template() to return either a "
            f"Tensor or a BatchEncoding object, but received an object "
            f"of type {type(input_tokens)} instead."
        )

    # generate() will fail with many different creative error messages if tokens aren't
    # on the right device.
    input_tokens = input_tokens.to(model.device)
    generate_input["input_tokens"] = input_tokens

    # The generate() method sometimes needs to know what is the integer ID
    # of the padding token, and for some reason this critical piece of information
    # isn't included in the serialized model. We get it from the tokenizer.
    # And of course some tokenizers don't set this parameter, in which case
    # we use the end of string token and hope for the best.
    pad_token_id = tokenizer.pad_token_id  # type: ignore[union-attr]
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id  # type: ignore[union-attr]
    if pad_token_id is None:
        # Raise an error here because the some branches of the generate
        # method won't complain about an invalid value of this parameter,
        # while others will raise a cryptic exception from deep within
        # their beam search code.
        raise ValueError(f"Couldn't figure out padding token for tokenizer {tokenizer}")
    generate_input["pad_token_id"] = pad_token_id

    # Make sure you specify this parameter explicitly, or you will have
    # a bad time.
    generate_input["eos_token_id"] = tokenizer.eos_token_id  # type: ignore[union-attr]

    other_input = {}

    if request.get("logprobs"):
        generate_input["output_scores"] = True

    if request.get("top_logprobs") is not None:
        # Transformers has no notion of top_logprobs. Pass it through so our own post-
        # processing code can deal with it on the other side.
        other_input["top_logprobs"] = request["top_logprobs"]

    if request.get("max_completion_tokens") is not None:
        generate_input["max_new_tokens"] = request["max_completion_tokens"]

    if (
        request.get("extra_body") is not None
        and request["extra_body"].get("structured_outputs") is not None
        and request["extra_body"].get("structured_outputs").get("json") is not None
    ):
        # Constrained decoding in Hugging Face requires using a third-party library
        # to create a callback function to be invoked from inside generate()
        with import_optional("llguidance"):
            # Third Party
            import llguidance
            import llguidance.hf

        if ll_tokenizer is None:
            # HF model components disagree on vocab size (resized embeddings, added
            # special tokens, etc.). Pass the maximum so the bitmask covers every
            # token id the model can emit. llguidance defaults to the tokenizer's
            # value when n_vocab is None, which can be smaller than model.vocab_size.
            n_vocab = max(tokenizer.vocab_size, len(tokenizer), model.vocab_size)  # type: ignore[arg-type]
            ll_tokenizer = llguidance.hf.from_tokenizer(tokenizer, n_vocab=n_vocab)  # type: ignore[arg-type]

        grammar = llguidance.LLMatcher.grammar_from_json_schema(
            request["extra_body"]["structured_outputs"]["json"]
        )
        logits_processor = _GuidanceLogitsProcessor(grammar, ll_tokenizer)

        # The "logits_processor" argument to generate() must be a list.
        generate_input["logits_processor"] = [logits_processor]  # type: ignore[assignment]

        if constrained_decoding_prefix is not None:
            # Some models generate boilerplate before getting to the place where the
            # logits processor should activate. Append that boilerplate to the prompt,
            # since the logits processor we just created will
            existing_tokens: torch.Tensor = generate_input["input_tokens"]
            addl_tokens = tokenizer(
                constrained_decoding_prefix, return_tensors="pt"
            ).to(model.device)["input_ids"]
            generate_input["input_tokens"] = torch.cat(  # type: ignore[assignment]
                [existing_tokens, addl_tokens],
                dim=1,  # type: ignore[list-item]
            )

    # Translate beam search parameters
    if request.get("temperature") is not None:
        if request["temperature"] == 0.0:
            # No beam search
            generate_input["do_sample"] = False
        else:
            # Beam search
            generate_input["do_sample"] = True
            generate_input["temperature"] = request["temperature"]

    if request.get("n") is not None:
        generate_input["num_return_sequences"] = request["n"]

    for param in ("top_k", "top_p"):
        if request.get(param) is not None:
            generate_input[param] = request[param]

    return generate_input, other_input


def generate_with_transformers(
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    generate_input: dict,
    other_input: dict,
) -> ChatCompletionResponse:
    """Call Transformers generate and get usable results.

    All the extra steps necessary to call the :func:`generate()` method of a
    Transformers model and get back usable results, rolled into a single function.

    There are quite a few extra steps.

    Args:
        tokenizer: HuggingFace tokenizer for the model, required at several stages
            of generation.
        model: Initialized HuggingFace model object.
        generate_input: Parameters to pass to the `generate()` method, usually
            produced by `chat_completion_request_to_transformers_inputs()`.
        other_input: Additional kwargs produced by
            `chat_completion_request_to_transformers_inputs()` for aspects of the
            original request that Transformers APIs don't handle natively.

    Returns:
        A chat completion response in OpenAI format.
    """
    with import_optional("torch"):
        # Third Party
        import torch

    input_tokens = generate_input["input_tokens"]
    generate_input = generate_input.copy()
    del generate_input["input_tokens"]

    generate_result = model.generate(input_ids=input_tokens, **generate_input)  # type: ignore[operator]

    # Result is a a 2D tensor of shape (num responses, prompt + max generated tokens)
    # containing tokens, plus a tuple of <max generated tokens> tensors of shape
    # (num beams, vocab size) containing scores.
    # This is of course not a usable format for downstream processing.
    # Start by stripping off the prompt, leaving us with a tensor of shape
    # (num responses, max generated tokens)
    num_prompt_tokens = input_tokens.shape[1]
    num_responses = generate_result.sequences.shape[0]
    generated_tokens = generate_result.sequences[:, num_prompt_tokens:]

    generated_scores = (
        None
        if generate_result.scores is None
        else (torch.stack(generate_result.scores).swapaxes(0, 1)[:num_responses])
    )

    # Iterate over the responses, stripping off EOS tokens
    choices = []
    for i in range(num_responses):
        response_tokens = generated_tokens[i]

        if tokenizer.eos_token_id in response_tokens:
            # Strip off everything after the first EOS token.
            # Pytorch syntax for finding the first EOS is a bit funky.
            eos_ix = (
                (response_tokens == tokenizer.eos_token_id)
                .nonzero(as_tuple=True)[0]
                .item()
            )
            response_tokens = response_tokens[:eos_ix]

        response_string = tokenizer.decode(response_tokens)

        # The decode() method doesn't return offsets.
        # The only supported API to get offsets is to retokenize the string and hope you
        # get back the same tokenization.
        # This supported API doesn't work reliably, so we fall back on the unsupported
        # method of pulling token lengths out of the tokenizer.
        # Transformers 5 changed the behavior of batch_decode() when fed a list of
        # individual token IDs, so we need to massage response_tokens into a format
        # that will produce the same result with version 5 and with older versions.
        list_of_singleton_lists = [[t] for t in response_tokens]
        ends = list(
            itertools.accumulate(
                [len(s) for s in tokenizer.batch_decode(list_of_singleton_lists)]
            )
        )
        begins = [0, *ends[:-1]]
        token_offsets = list(zip(begins, ends, strict=True))

        if generated_scores is None:
            logprobs_content = None
        else:
            response_scores = generated_scores[i]

            # Scores come back as raw logits. You need to decode them to produce
            # logprobs. For consistency with the OpenAI output format, we need to
            # decode twice: Once to get the probability of the returned token and a
            # second time to get the top k logprobs. As with the OpenAI APIs, the
            # returned token may or may not be included in the top k results.
            all_logprobs = torch.log_softmax(response_scores.to(torch.float32), 1)
            chosen_token_logprobs = [
                all_logprobs[token_ix][response_tokens[token_ix]].item()
                for token_ix in range(len(response_tokens))
            ]
            assert isinstance(response_string, str)
            token_strings = [response_string[begin:end] for begin, end in token_offsets]
            token_bytes = [list(s.encode("utf-8")) for s in token_strings]

            # Transformers has no notion of top-k logprobs, so the parameter that
            # triggers that post-processing is passed via other_input.
            if "top_logprobs" not in other_input:
                top_logprobs: list = [[] for _ in range(len(token_strings))]
            else:  # if "top_logprobs" in other_input:
                top_k_values, top_k_indices = torch.topk(
                    torch.nan_to_num(all_logprobs, float("-inf")),
                    other_input["top_logprobs"],
                )
                top_k_token_strs: list[list[str]] = [
                    [str(tokenizer.decode(t)) for t in row_i] for row_i in top_k_indices
                ]
                top_logprobs = [
                    [
                        {
                            "token": s,
                            "bytes": list(s.encode("utf8")),
                            "logprob": lp.item(),
                        }
                        for s, lp in zip(strs, lps, strict=True)
                    ]
                    for strs, lps in zip(top_k_token_strs, top_k_values, strict=True)
                ]

            logprobs_content = [
                {
                    "token": token_strings[i],
                    "bytes": token_bytes[i],
                    "logprob": chosen_token_logprobs[i],
                    "top_logprobs": top_logprobs[i],
                }
                for i in range(len(response_tokens))
            ]

        response_choice_value = {
            "index": i,
            "message": {"content": response_string, "role": "assistant"},
        }
        if logprobs_content is not None:
            response_choice_value["logprobs"] = {"content": logprobs_content}
        response_choice = ChatCompletionResponseChoice.model_validate(
            response_choice_value
        )
        choices.append(response_choice)

    return ChatCompletionResponse(choices=choices)
