"""Huggingface implementations for IBM's "starter pack" of Activated LoRAs."""

import asyncio
import functools
from copy import deepcopy

import granite_common
import torch
from transformers.generation.utils import GenerateDecoderOnlyOutput

from mellea.backends.huggingface import HFAlora, HFAloraCacheInfo, LocalHFBackend
from mellea.backends.types import ModelOption
from mellea.helpers.async_helpers import send_to_queue
from mellea.helpers.fancy_logger import FancyLogger
from mellea.stdlib.base import GenerateType, ModelOutputThunk


class HFConstraintAlora(HFAlora):
    """The Requirement Checking ALora for Granite checks if the specified requirement was satisfied by the most recent model generation. Only one requirement is checked at a time.

    Currently supports [Granite 3.2 8B](https://huggingface.co/ibm-granite/granite-3.2-8b-alora-requirement-check) and [Granite 3.3 8B](https://huggingface.co/ibm-granite/granite-3.3-8b-alora-requirement-check) by default.
    """

    def __init__(
        self,
        name: str,
        path_or_model_id: str,
        generation_prompt: str,
        backend: LocalHFBackend,
        *,
        constraint_prompt: str | None = None,
        include_constraint_in_alora_offset: bool = False,
    ):
        """Initialize after checking that the backend is correct.

        Args:
            name: name of the alora.
            path_or_model_id: huggingface path or model id.
            generation_prompt: the prompt required to activate the aLoRa.
            backend: a LocalHFBackend that this alora is attached to.
            constraint_prompt: a template that the constraint can be interpolated into; can only have a single `{}` slot.
            include_constraint_in_alora_offset: whether to include the constraint prompt in the alora offset.
        """
        super().__init__(name, path_or_model_id, generation_prompt, backend)

        # Maintain default behavior.
        if constraint_prompt is None:
            constraint_prompt = "\nRequirement: {}<|end_of_text|>\n"

        self._constraint_prompt = constraint_prompt
        self._include_constraint_in_alora_offset = include_constraint_in_alora_offset

        # We do a lot of logging for ALoras because this is an experimental feature. Maybe we should tag these log messages?
        self._logger = FancyLogger.get_logger()

    def generate_using_strings(
        self,
        input: str,
        response: str,
        constraint: str,
        force_yn: bool = True,
        stream: bool = False,
    ) -> ModelOutputThunk:
        """Generates a constraint response from the ALora. Must be run in a running event loop."""
        # Go ahead and do runtime type-checking because passing CBlocks into this function is a common error.
        assert type(input) is str
        assert type(response) is str
        assert type(constraint) is str
        self._backend._model.set_adapter(self.name)
        cache_hit = self._backend.cache_get(response)

        if stream:
            self._logger.warning(
                "`HFConstraintAlora` cannot stream output; defaulting to non-streaming approach."
            )

        generate_kwargs = {}
        if cache_hit:
            self._logger.debug(
                f"using cache for alora {self.__class__} and response '{response}'"
            )
            generate_kwargs["past_key_values"] = deepcopy(cache_hit.kv_cache)
            input_combined = self._generate_using_cache(cache_hit, constraint, force_yn)

        else:
            self._logger.debug(
                f"not using cache for alora {self.__class__} and response '{response}'"
            )
            input_combined = self._generate_not_using_cache(
                input, response, constraint, force_yn
            )

        if not self._include_constraint_in_alora_offset:
            alora_offsets = [self._generation_prompt_tokens["input_ids"].shape[1] - 1]
        else:
            # Get the constraint tokens separately so that we can calculate the alora offsets.
            constraint_tokens = self._backend._tokenizer(
                self._constraint_prompt.format(constraint), return_tensors="pt"
            )

            alora_offsets = [
                constraint_tokens["input_ids"].shape[1]
                + self._generation_prompt_tokens["input_ids"].shape[1]
                - 2
            ]

        chat_response = asyncio.to_thread(
            self._backend._model.generate,  # type: ignore
            input_combined["input_ids"],
            attention_mask=input_combined["attention_mask"],
            max_new_tokens=1,
            return_dict_in_generate=True,
            alora_offsets=alora_offsets,
            output_scores=True,
            **generate_kwargs,
        )

        output = ModelOutputThunk(None)
        output._meta["alora_name"] = self.name

        output._process = functools.partial(
            processing,
            backend=self._backend,
            force_yn=force_yn,
            gen_prompt=self._generation_prompt,
        )
        output._post_process = functools.partial(post_processing, backend=self._backend)

        try:
            # To support lazy computation, will need to remove this create_task and store just the unexecuted coroutine.
            # We can also support synchronous calls by adding a flag and changing this ._generate function.

            # This function should always be called from a running event loop so we don't have to worry about
            # scheduling the task to a specific event loop here.
            output._generate = asyncio.create_task(
                send_to_queue(chat_response, output._async_queue)  # type: ignore
            )
            output._generate_type = GenerateType.ASYNC
        except RuntimeError as e:
            # Most likely cause is running this function without an event loop present.
            raise e

        return output

    def _generate_using_cache(
        self, cache_hit: HFAloraCacheInfo, constraint: str, force_yn: bool
    ) -> dict:
        """Returns the input object used for generation."""
        # Must tokenize the constraint here since the requirement isn't known at initialization.
        constraint_tokens = self._backend._tokenizer(
            self._constraint_prompt.format(constraint), return_tensors="pt"
        )

        input_combined = {
            "input_ids": torch.cat(
                [
                    cache_hit.merged_token_ids.unsqueeze(0),
                    constraint_tokens["input_ids"],
                    self._generation_prompt_tokens["input_ids"],
                ],
                dim=1,
            ),
            "attention_mask": torch.cat(
                [
                    cache_hit.merged_attention.unsqueeze(0),
                    constraint_tokens["attention_mask"],
                    self._generation_prompt_tokens["attention_mask"],
                ],
                dim=1,
            ),
        }

        self._logger.debug(
            f"Prompt for cached aLoRA({self.name}):\n {self._backend._tokenizer.decode(input_combined['input_ids'][0])}"
        )

        return input_combined

    def _generate_not_using_cache(
        self, input: str, response: str, constraint: str, force_yn: bool
    ) -> dict:
        """Returns the input object used for generation."""
        # Params aren't needed when just getting the backend args.
        backend_model_opts = self._backend._simplify_and_merge(None)
        sys_prompt = backend_model_opts.get(ModelOption.SYSTEM_PROMPT, None)

        chat = [
            *([{"role": "system", "content": sys_prompt}] if sys_prompt else []),
            {"role": "user", "content": input},
            {"role": "assistant", "content": response},
        ]

        templatized = self._backend._tokenizer.apply_chat_template(chat, tokenize=False)
        assert type(templatized) is str

        # Must tokenize the constraint here since the requirement isn't known at initialization.
        templatized = templatized + self._constraint_prompt.format(constraint)

        tokenized = self._backend._tokenizer(templatized, return_tensors="pt")
        input_combined = {
            "input_ids": torch.cat(
                [tokenized["input_ids"], self._generation_prompt_tokens["input_ids"]],
                dim=1,
            ),
            "attention_mask": torch.cat(
                [
                    tokenized["attention_mask"],
                    self._generation_prompt_tokens["attention_mask"],
                ],
                dim=1,
            ),
        }

        self._logger.debug(
            f"Prompt for non-cached aLoRA({self.name}):\n{self._backend._tokenizer.decode(input_combined['input_ids'][0])}"
        )

        return input_combined


async def processing(
    mot: ModelOutputThunk,
    chunk: GenerateDecoderOnlyOutput,
    backend: LocalHFBackend,
    force_yn: bool,
    gen_prompt: str,
):
    """Called to process the incoming chunks."""
    if mot._underlying_value is None:
        mot._underlying_value = ""

    # Don't support async for HFConstraintAlora. Means we can process the output here.
    assert isinstance(chunk, GenerateDecoderOnlyOutput)

    if force_yn:
        last_logits = chunk.scores[-1].squeeze(0)  # type: ignore
        token_Y = backend._tokenizer("Y", add_special_tokens=False)["input_ids"][0]  # type: ignore
        token_N = backend._tokenizer("N", add_special_tokens=False)["input_ids"][0]  # type: ignore
        logit_Y = last_logits[token_Y].item()
        logit_N = last_logits[token_N].item()
        mot._underlying_value = "Y" if logit_Y > logit_N else "N"
    else:
        output_text = backend._tokenizer.decode(chunk.sequences[0])
        constraint_satisfied = output_text.split(gen_prompt)[-1]
        mot._underlying_value = constraint_satisfied[
            0
        ]  # Grab the first char of the str.


async def post_processing(mot: ModelOutputThunk, backend: LocalHFBackend):
    """Called after all data has been received."""
    backend.formatter.parse(mot._action, mot)  # type: ignore


def add_granite_aloras(backend: LocalHFBackend):
    """Adds the IBM Granite "starter pack" ALoras to a backend."""
    if backend._hf_model_id == "ibm-granite/granite-3.2-8b-instruct":
        backend.add_alora(
            HFConstraintAlora(
                name="constraint",
                path_or_model_id="ibm-granite/granite-3.2-8b-alora-requirement-check",
                generation_prompt="<|start_of_role|>check_requirement<|end_of_role|>",
                backend=backend,
                constraint_prompt="\nRequirement: {}<|end_of_text|>\n",
                include_constraint_in_alora_offset=False,
            )
        )
    elif backend._hf_model_id == "ibm-granite/granite-3.3-8b-instruct":
        backend.add_alora(
            HFConstraintAlora(
                name="constraint",
                path_or_model_id="ibm-granite/granite-3.3-8b-alora-requirement-check",
                generation_prompt="<|start_of_role|>check_requirement<|end_of_role|>",
                backend=backend,
                constraint_prompt="\n<|start_of_role|>requirement<|end_of_role|>{}<|end_of_text|>\n",
                include_constraint_in_alora_offset=True,
            )
        )
    else:
        raise ValueError(
            f"cannot add_granite_aloras to unknown huggingface model_id / backend: {backend._hf_model_id}"
        )


class HFIntrinsicAlora(HFAlora):
    def __init__(
        self, name: str, backend: LocalHFBackend, base_model_name: str | None = None
    ):
        # TODO: JAL. May need to change HFAlora to not need a generation prompt by default...

        if base_model_name is None:
            # TODO: JAL. Make sure all hf model ids fit this way.
            base_model_name = backend._hf_model_id.split("/")[1]

        # TODO: JAL. figure out if we want to download files here or when adding the alora...
        # Need to do all the path determination at init?
        path = str(
            granite_common.intrinsics.util.obtain_lora(
                name, base_model_name, alora=True
            )
        )

        super().__init__(name, path, "", backend)

        # We do a lot of logging for ALoras because this is an experimental feature. Maybe we should tag these log messages?
        self._logger = FancyLogger.get_logger()

    def generate_using_strings(self, *args, **kwargs) -> ModelOutputThunk:
        raise NotImplementedError()
