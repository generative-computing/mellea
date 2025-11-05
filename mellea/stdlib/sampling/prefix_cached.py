"""Sampling Strategy that uses prefix caching idea based on two turn chats."""

from collections.abc import Awaitable, Callable

from mellea.backends import Backend, BaseModelSubclass, ModelOption
from mellea.stdlib.base import ChatContext, Component, Context, ContextTurn
from mellea.stdlib.chat import Message
from mellea.stdlib.requirement import Requirement, ValidationResult
from mellea.stdlib.sampling import RejectionSamplingStrategy, SamplingResult


class RejectionSamplingStrategyWithPrefix(RejectionSamplingStrategy):
    """Rejection Sampling class that uses the last turn as prefix cache for requirement checking."""

    async def sample(
        self,
        action: Component,
        context: Context,
        backend: Backend,
        requirements: list[Requirement] | None,
        *,
        validation_ctx: Context | None = None,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
        show_progress: bool = True,
    ) -> SamplingResult:
        """Sample method inherited from RejectionSamplingStrategy."""
        reqs: list[Requirement] = []
        if self.requirements is not None:
            reqs += self.requirements
        elif requirements is not None:
            reqs += requirements
        reqs = list(set(reqs))

        def make_val(
            req_string: str,
        ) -> Callable[[Context], Awaitable[ValidationResult]]:
            async def validate_agentic(ctx: Context) -> ValidationResult:
                lt = ctx.last_turn()
                assert isinstance(lt, ContextTurn)
                assert lt.model_input is not None
                assert lt.output is not None

                chat_ctx = ChatContext()
                chat_ctx = chat_ctx.add(lt.model_input)
                chat_ctx = chat_ctx.add(lt.output)

                action = Message(
                    role="user",
                    content=f"Does the output fulfill the requirement? Answer only with yes or no.  Requirement: '{req_string}'",
                )

                llm_as_a_judge_result, _ = backend.generate_from_context(
                    action,
                    chat_ctx,
                    format=format,
                    model_options={ModelOption.MAX_NEW_TOKENS: 10},
                )
                value = await llm_as_a_judge_result.avalue()

                return ValidationResult(
                    result=value.lower().startswith("yes"),
                    reason=value,
                    thunk=llm_as_a_judge_result,
                )

            return validate_agentic

        for req in reqs:
            if req.validation_fn is None:
                req.validation_fn = make_val(str(req.description))

        res = await super().sample(
            action=action,
            context=context,
            backend=backend,
            requirements=reqs,
            validation_ctx=validation_ctx,
            format=format,
            model_options=model_options,
            tool_calls=tool_calls,
            show_progress=show_progress,
        )
        return res
