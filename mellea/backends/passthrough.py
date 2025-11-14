import asyncio
import inspect
from collections.abc import Callable, Coroutine
from typing import Any, cast

from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

import mellea.backends.model_ids as model_ids
from mellea.backends import BaseModelSubclass
from mellea.backends.formatter import Formatter, FormatterBackend, TemplateFormatter
from mellea.backends.openai import OpenAIBackend
from mellea.backends.types import ModelOption
from mellea.helpers.async_helpers import send_to_queue
from mellea.helpers.event_loop_helper import _run_async_in_thread
from mellea.stdlib.base import (
    CBlock,
    Component,
    Context,
    GenerateType,
    ModelOutputThunk,
)
from mellea.stdlib.chat import Message

# TODO: JAL. Maybe this should be called a wrapper backend.
# TODO: JAL. Maybe more customizability should be exposed...
#            input / output types
#            processing into model-output-thunks -> could be a helper function for this, you point us to where the message is in the response type
class PassthroughBackend(FormatterBackend):
    def __init__(
        self,
        # TODO: JAL. This could probably return anything; just have to process it.
        # TODO: JAL. Figure out how to signal kwargs in the typing; moresoe than Any
        # TODO: JAL. Make this into two params; one that takes openai messages for from context
        #            and one that takes strings for generate_from_raw
        generate: Callable[
            [list[dict], Any], Coroutine[None, None, ChatCompletion] | ChatCompletion
        ]
        # | Callable[
        #     [list[dict], Any], ChatCompletion
        # ]
        | None,
        generate_raw: Callable[[list[str], Any], Coroutine[None, None, list[str]] | list[str]] | None,
        *,
        model_id: str | model_ids.ModelIdentifier | None = None,
        formatter: Formatter | None = None,
        model_options: dict | None = None,
    ):
        """Initializes a backend that uses your generation function.

        Note: If None is provided for either generate or generate_raw, you will not be able to use backend.generate_from_context or backend.generate_from_raw respectively.

        Both generate and generate_raw must accept a positional arg representing either the conversation or the raw string.

        Args:
            generate: a (sync or async) function that takes a list of openai messages and returns a chat completion response
            generate_raw: a (sync or async) function that takes a string representing the current context and returns a string
            model_id: (optional) if provided, will be used to create a formatter; defaults to 'ibm/granite4:micro'
            formatter: (optional) if provided, will be used as the formatter for this backend; if None, will use a TemplateFormatter
            model_options: (optional) additional kwargs to pass to the generate function; unlike other backends, we do not do any special processing of these options.
        """
        if model_id is None:
            model_id = model_ids.IBM_GRANITE_4_MICRO_3B

        if formatter is None:
            formatter = TemplateFormatter(model_id=model_id)

        self.generate = generate
        self.generate_raw = generate_raw

        self.model_id = model_id
        self.formatter = formatter
        self.model_options = model_options

    def _simplify_and_merge(
        self, model_options: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Merges the model_options this backend was initialized with and those passed into the specific generate call.

        Rules:
        - When merging, the keys/values from the dictionary passed into this function take precedence.

        Args:
            model_options: the model_options for this call

        Returns:
            a new dict
        """
        backend_model_opts = (
            self.model_options if self.model_options is not None else {}
        )
        if model_options is None:
            return backend_model_opts

        return ModelOption.merge_model_options(backend_model_opts, model_options)

    def generate_from_context(
        self,
        action: Component | CBlock,
        ctx: Context,
        *,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> tuple[ModelOutputThunk, Context]:
        """Generate from an action and a context.

        Args:
            action: the next action to generate with
            ctx: the context used for this request
            format: ignored for this backend
            model_options: additional kwargs used for generation; not processed specially for this backend
            tool_calls: ignored for this backend
        """
        if self.generate is None:
            raise RuntimeError(
                "cannot call `generate_from_context` on a PassthroughBackend initialized with `generate=None`"
            )

        model_opts = self._simplify_and_merge(model_options)

        linearized_context = ctx.view_for_generation()
        assert linearized_context is not None, (
            "Cannot generate from a non-linear context in a FormatterBackend."
        )
        linearized_context.append(action)
        messages: list[Message] = self.formatter.to_chat_messages(linearized_context)
        conversation: list[dict] = [
            OpenAIBackend.message_to_openai_message(m) for m in messages
        ]

        is_sync = inspect.iscoroutinefunction(self.generate)
        if is_sync:
            chat_response = asyncio.to_thread(
                self.generate,
                conversation,
                **model_opts,  # TODO: JAL. Figure out this typing.
            )
        else:
            chat_response = self.generate(
                conversation,
                **model_opts,  # TODO: JAL. Figure out this typing.
            )

        output = ModelOutputThunk(None)
        output._context = linearized_context
        output._action = action
        output._model_options = model_opts

        # Processing functions only pass the ModelOutputThunk (and current chunk of response). Bind the other vars necessary for
        # each processing step.
        output._process = self.processing
        output._post_process = self.post_processing

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

        return output, ctx.add(action).add(output)

    async def processing(self, mot: ModelOutputThunk, chunk: ChatCompletion):
        """Called during generation to add information from a single ChatCompletion to the ModelOutputThunk."""
        # TODO: JAL. This should be more accepting in the type of response it takes. ie all dicts or the actual object...
        #            look at watsonx; I think its a dict version of this struct...
        if mot._thinking is None:
            mot._thinking = ""
        if mot._underlying_value is None:
            mot._underlying_value = ""

        message = chunk.choices[0].message

        if hasattr(message, "reasoning_content"):
            thinking_chunk = message.reasoning_content  # type: ignore
            if thinking_chunk is not None:
                mot._thinking += thinking_chunk

        content_chunk = message.content
        if content_chunk is not None:
            mot._underlying_value += content_chunk

        mot._meta["oai_chat_response"] = chunk.choices[0].model_dump()

    async def post_processing(self, mot: ModelOutputThunk):
        """Intentionally do nothing here."""
        return

    # TODO: JAL. There's a meta question here of whether this should operate over str -> CompletionResponse or str -> str.
    def generate_from_raw(
        self,
        actions: list[Component | CBlock],
        ctx: Context,
        *,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> list[ModelOutputThunk]:
        if self.generate_raw is None:
            raise RuntimeError(
                "cannot call `generate_from_raw` on a PassthroughBackend initialized with `generate_raw=None`"
            )

        model_opts = self._simplify_and_merge(model_options)
        prompts = [self.formatter.print(action) for action in actions]
        is_sync = inspect.iscoroutinefunction(self.generate)
        # TODO: JAL. Change all this logic once this is made async.
        if is_sync:
            chat_response = self.generate_raw(
                prompts,
                **model_opts,  # TODO: JAL. Figure out this typing.
            )
        else:
            chat_response = _run_async_in_thread(
                self.generate_raw(
                    prompts,
                    **model_opts,  # TODO: JAL. Figure out this typing.
                )
            )

        chat_response = cast(list[str], chat_response)
        results = []
        for response in chat_response:
            # TODO: JAL. Need to do more here. and fill out remaining fields.
            results.append(ModelOutputThunk(response))

        return results
