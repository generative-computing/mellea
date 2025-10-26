"""OpenAI implementations for IBM's "starter pack" of Activated LoRAs."""

import asyncio
import functools
from collections.abc import Coroutine
from typing import Any

import openai
from openai.types.completion import Completion

from mellea.backends.aloras import Alora
from mellea.backends.openai import OpenAIAlora, OpenAIBackend
from mellea.backends.types import ModelOption
from mellea.helpers.async_helpers import send_to_queue
from mellea.helpers.fancy_logger import FancyLogger
from mellea.stdlib.base import GenerateType, ModelOutputThunk


# TODO: JAL. this should be made into a generic adapter interface...?
class OpenAIIntrinsicAlora(OpenAIAlora):
    """The [Requirement Checking ALora for Granite 3.2 8B](https://huggingface.co/ibm-granite/granite-3.2-8b-alora-requirement-check) checks if the specified requirement was satisfied by the most recent model generation. Only one requirement is checked at a time."""

    def __init__(
        self, name: str, path: str, generation_prompt: str, backend: OpenAIBackend
    ):
        """Initialize after checking that the backend is correct."""
        super().__init__(name, path, generation_prompt, backend)
        # We do a lot of logging for ALoras because this is an experimental feature. Maybe we should tag these log messages?
        self._logger = FancyLogger.get_logger()
        # TODO: JAL. Move the adapter loading logic to the init?... or see where we want to make sure they are downloaded...

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

        # Params aren't needed when just getting the backend args.
        backend_model_opts = self._backend._simplify_and_merge(None, False)
        sys_prompt = backend_model_opts.get(ModelOption.SYSTEM_PROMPT, None)

        chat = [
            *([{"role": "system", "content": sys_prompt}] if sys_prompt else []),
            {"role": "user", "content": input},
            {"role": "assistant", "content": response},
        ]

        prompt = self._backend.apply_chat_template(chat)
        prompt += f"\nRequirement: {constraint}<|end_of_text|>\n"  # type: ignore
        prompt += self._generation_prompt

        self._logger.debug(f"Prompt for non-cached aLoRA({self.name}):\n{prompt}")

        force_yn_args = {}
        if force_yn:
            assert hasattr(self._backend, "_tokenizer")
            token_Y = self._backend._tokenizer("Y", add_special_tokens=False)[
                "input_ids"
            ][0]  # type: ignore
            token_N = self._backend._tokenizer("N", add_special_tokens=False)[
                "input_ids"
            ][0]  # type: ignore

            force_yn_args["logit_bias"] = {str(token_Y): 100, str(token_N): 100}

        chat_response: Coroutine[
            Any, Any, openai.AsyncStream[Completion] | Completion
        ] = self._backend._async_client.completions.create(
            model=self.name,
            prompt=prompt,
            max_tokens=1,
            n=1,
            stream=stream,
            **force_yn_args,
        )  # type: ignore

        output = ModelOutputThunk(None)
        output._meta["alora_name"] = self.name

        output._process = processing
        output._post_process = functools.partial(post_processing, self._backend)

        try:
            # To support lazy computation, will need to remove this create_task and store just the unexecuted coroutine.
            # We can also support synchronous calls by adding a flag and changing this ._generate function.

            # This function should always be called from a running event loop so we don't have to worry about
            # scheduling the task to a specific event loop here.
            output._generate = asyncio.create_task(
                send_to_queue(chat_response, output._async_queue)
            )
            output._generate_type = GenerateType.ASYNC
        except RuntimeError as e:
            # Most likely cause is running this function without an event loop present
            raise e

        return output


async def processing(mot: ModelOutputThunk, chunk: Completion):
    """Called to process the incoming chunks."""
    if mot._underlying_value is None:
        mot._underlying_value = ""
    mot._underlying_value += chunk.choices[0].text


async def post_processing(backend: OpenAIBackend, mot: ModelOutputThunk):
    """Called after all data has been received."""
    backend.formatter.parse(mot._action, mot)  # type: ignore


def add_granite_aloras(backend: OpenAIBackend):
    """Adds the IBM Granite "starter pack" ALoras to a backend."""
    if backend._hf_model_id == "ibm-granite/granite-3.2-8b-instruct":
        backend.add_alora(
            OpenAIIntrinsicAlora(
                name="constraint",
                path="ibm-granite/granite-3.2-8b-alora-requirement-check",
                generation_prompt="<|start_of_role|>check_requirement<|end_of_role|>",
                backend=backend,
            )
        )
    elif backend._hf_model_id == "ibm-granite/granite-3.3-8b-instruct":
        backend.add_alora(
            OpenAIIntrinsicAlora(
                name="constraint",
                path="ibm-granite/granite-3.3-8b-alora-requirement-check",
                generation_prompt="<|start_of_role|>check_requirement<|end_of_role|>",
                backend=backend,
            )
        )
    else:
        raise ValueError(
            f"cannot add_granite_aloras to unknown huggingface model_id / backend: {backend._hf_model_id}"
        )
