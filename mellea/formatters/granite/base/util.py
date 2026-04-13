# SPDX-License-Identifier: Apache-2.0

"""Common utility functions for the library and tests."""

# Standard
from __future__ import annotations

import contextlib
import itertools
import json
import logging
import os
import re
import uuid
from typing import TYPE_CHECKING

# Third Party
import pydantic

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

# First Party
from .types import ChatCompletionResponse, ChatCompletionResponseChoice


@contextlib.contextmanager
def import_optional(extra_name: str):
    """Handle optional imports.

    Args:
        extra_name: Package extra to suggest in the install hint
            (e.g. ``pip install mellea[extra_name]``).
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


def find_substring_in_text(substring: str, text: str) -> list[dict]:
    """Find all substring matches in text.

    Given two strings - substring and text - find and return all
    matches of substring within text. For each match return its begin and end index.

    Args:
        substring: The string to search for.
        text: The string to search within.

    Returns:
        List of dicts with ``begin_idx`` and ``end_idx`` for each match found.
    """
    span_matches = []

    matches_iter = re.finditer(re.escape(substring), text)
    for match in matches_iter:
        span_matches.append({"begin_idx": match.start(), "end_idx": match.end()})

    return span_matches


def random_uuid() -> str:
    """Generate a random UUID string.

    Returns:
        Hexadecimal UUID string suitable for use as a unique identifier.
    """
    return str(uuid.uuid4())


def load_transformers_lora(local_or_remote_path: str) -> tuple:
    """Load transformers LoRA model.

    AutoModelForCausalLM.from_pretrained() is supposed to auto-load base models if you
    pass it a LoRA adapter's config, but that auto-loading is very broken as of 8/2025.
    Workaround powers activate!

    Only works if ``transformers`` and ``peft`` are installed.

    Args:
        local_or_remote_path: Local directory path of the LoRA adapter.

    Returns:
        Tuple of ``(model, tokenizer)`` where ``model`` is the loaded LoRA model and
        ``tokenizer`` is the corresponding HuggingFace tokenizer.

    Raises:
        ImportError: If ``peft`` or ``transformers`` packages are not installed.
        NotImplementedError: If ``local_or_remote_path`` does not exist locally
            (remote loading from the Hugging Face Hub is not yet implemented).
    """
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
    base_model = transformers.AutoModelForCausalLM.from_pretrained(base_model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(base_model_name)
    model = peft.PeftModel.from_pretrained(base_model, local_model_dir)
    return model, tokenizer


def chat_completion_request_to_transformers_inputs(
    request: dict,
    tokenizer: PreTrainedTokenizerBase | None = None,
    model: PreTrainedModel | None = None,
    constrained_decoding_prefix: str | None = None,
) -> tuple[dict, dict]:
    """Translate an OpenAI-style chat completion request.

    Translate an OpenAI-style chat completion request into an input for a Transformers
    ``generate()`` call.

    Args:
        request: Request as parsed JSON or equivalent dataclass.
        tokenizer: HuggingFace tokenizer for the model. Only required if the request
            uses constrained decoding.
        model: HuggingFace model object. Only required if the request uses constrained
            decoding.
        constrained_decoding_prefix: Optional generation prefix to append to the prompt.

    Returns:
        Tuple of ``(generate_input, other_input)`` where ``generate_input`` contains
        kwargs to pass directly to ``generate()`` and ``other_input`` contains
        additional parameters for ``generate_with_transformers``.

    Raises:
        ImportError: If ``torch``, ``transformers``, or ``xgrammar`` packages
            are not installed (the latter only when constrained decoding is used).
        TypeError: If ``tokenizer.apply_chat_template()`` returns an unexpected type.
        ValueError: If padding or end-of-sequence token IDs cannot be determined
            from the tokenizer, or if a constrained-decoding request is made
            without passing a ``tokenizer`` or ``model`` argument.
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
    input_tokens = input_tokens.to(model.device)  # type: ignore[union-attr]
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
        with import_optional("xgrammar"):
            # Third Party
            import xgrammar as xgr  # type: ignore[import-not-found]
        if tokenizer is None:
            raise ValueError(
                "Request specifies constrained decoding, but no "
                "tokenizer object was passed to this function."
            )
        if model is None:
            raise ValueError(
                "Request specifies constrained decoding, but no "
                "tokenizer object was passed to this function."
            )

        # Different parts of a Hugging Face model will have different opinions about
        # the number of tokens in the tokenizer's vocabulary, because of course they do.
        # Gather together all the possibilities and pick the biggest one.
        vocab_size = max(tokenizer.vocab_size, len(tokenizer), model.vocab_size)

        tokenizer_info = xgr.TokenizerInfo.from_huggingface(
            tokenizer, vocab_size=vocab_size
        )
        grammar_compiler = xgr.GrammarCompiler(tokenizer_info)
        compiled_grammar = grammar_compiler.compile_json_schema(
            request["extra_body"]["structured_outputs"]["json"]
        )
        logits_processor = xgr.contrib.hf.LogitsProcessor(compiled_grammar)

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
        generate_input: Parameters to pass to the ``generate()`` method, usually
            produced by ``chat_completion_request_to_transformers_inputs()``.
        other_input: Additional kwargs produced by
            ``chat_completion_request_to_transformers_inputs()`` for aspects of the
            original request that Transformers APIs don't handle natively.

    Returns:
        A chat completion response in OpenAI format.
    """
    with import_optional("torch"):
        # Third Party
        import torch

    # Input tokens must be passed to generate() as a positional argument, not a named
    # argument.
    input_tokens = generate_input["input_tokens"]
    generate_input = generate_input.copy()
    del generate_input["input_tokens"]

    generate_result = model.generate(input_tokens, **generate_input)  # type: ignore[operator]

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
                top_k_token_strs = [
                    [tokenizer.decode(t) for t in row_i] for row_i in top_k_indices
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
