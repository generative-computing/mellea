"""A method to generate outputs based on python functions and a Generative Slot function."""

import abc
import functools
import inspect
from collections.abc import Awaitable, Callable, Coroutine
from copy import deepcopy
from dataclasses import dataclass
from typing import (
    Any,
    Generic,
    ParamSpec,
    TypedDict,
    TypeVar,
    Union,
    get_type_hints,
    overload,
)

from pydantic import BaseModel, Field, create_model

import mellea.stdlib.funcs as mfuncs
from mellea.backends import Backend
from mellea.stdlib.base import Component, Context, TemplateRepresentation
from mellea.stdlib.requirement import Requirement, reqify
from mellea.stdlib.sampling.base import RejectionSamplingStrategy
from mellea.stdlib.sampling.types import SamplingStrategy
from mellea.stdlib.session import MelleaSession

P = ParamSpec("P")
R = TypeVar("R")


class FunctionResponse(BaseModel, Generic[R]):
    """Generic base class for function response formats."""

    result: R = Field(description="The function result")


def create_response_format(func: Callable[..., R]) -> type[FunctionResponse[R]]:
    """Create a Pydantic response format class for a given function.

    Args:
        func: A function with exactly one argument

    Returns:
        A Pydantic model class that inherits from FunctionResponse[T]
    """
    type_hints = get_type_hints(func)
    return_type = type_hints.get("return", Any)

    class_name = f"{func.__name__.replace('_', ' ').title().replace(' ', '')}Response"

    ResponseModel = create_model(
        class_name,
        result=(return_type, Field(description=f"Result of {func.__name__}")),
        __base__=FunctionResponse[return_type],  # type: ignore
    )

    return ResponseModel


class FunctionDict(TypedDict):
    """Return Type for a Function Component."""

    name: str
    signature: str
    docstring: str | None


class ArgumentDict(TypedDict):
    """Return Type for a Argument Component."""

    name: str | None
    annotation: str | None
    value: str | None


def describe_function(func: Callable) -> FunctionDict:
    """Generates a FunctionDict given a function.

    Args:
        func : Callable function that needs to be passed to generative slot.

    Returns:
        FunctionDict: Function dict of the passed function.
    """
    return {
        "name": func.__name__,
        "signature": str(inspect.signature(func)),
        "docstring": inspect.getdoc(func),
    }


def get_annotation(func: Callable, key: str, val: Any) -> str:
    """Returns a annotated list of arguments for a given function and list of arguments.

    Args:
        func : Callable Function
        key : Arg keys
        val : Arg Values

    Returns:
        str: An annotated string for a given func.
    """
    sig = inspect.signature(func)
    param = sig.parameters.get(key)
    if param and param.annotation is not inspect.Parameter.empty:
        return str(param.annotation)
    return str(type(val))


def bind_function_arguments(
    func: Callable[P, R], *args: P.args, **kwargs: P.kwargs
) -> dict[str, Any]:
    """Bind arguments to function parameters and return as dictionary.

    Args:
        func: The function to bind arguments for.
        *args: Positional arguments to bind.
        **kwargs: Keyword arguments to bind.

    Returns:
        Dictionary mapping parameter names to bound values with defaults applied.
    """
    signature = inspect.signature(func)
    bound_arguments = signature.bind(*args, **kwargs)
    bound_arguments.apply_defaults()
    return dict(bound_arguments.arguments)


class Argument:
    """An Argument Component."""

    def __init__(
        self,
        annotation: str | None = None,
        name: str | None = None,
        value: str | None = None,
    ):
        """An Argument Component."""
        self._argument_dict: ArgumentDict = {
            "name": name,
            "annotation": annotation,
            "value": value,
        }


class Function:
    """A Function Component."""

    def __init__(self, func: Callable):
        """A Function Component."""
        self._func: Callable = func
        self._function_dict: FunctionDict = describe_function(func)


@dataclass
class ExtractedKwargs:
    """Used to extract the mellea args and original function args."""

    f_kwargs: dict
    session: MelleaSession | None = None
    context: Context | None = None
    backend: Backend | None = None
    model_options: dict | None = None
    requirements: list[Requirement | str] | None = None
    strategy: SamplingStrategy | None = None

    def __init__(self):
        """Used to extract the mellea args and original function args."""
        self.f_kwargs = {}


class GenerativeSlot(Component, Generic[P, R]):
    """A generative slot component."""

    def __init__(self, func: Callable[P, R]):
        """A generative slot function that converts a given `func` to a generative slot.

        Args:
            func: A callable function
        """
        self._function = Function(func)
        self._arguments: list[Argument] = []
        functools.update_wrapper(self, func)

    @abc.abstractmethod
    def __call__(self, *args, **kwargs) -> tuple[R, Context] | R:
        """Call the generative slot.

        Args:
            m: MelleaSession: A mellea session (optional; must set context and backend if None)
            context: the Context object (optional; session must be set if None)
            backend: the backend used for generation (optional: session must be set if None)
            model_options: Model options to pass to the backend.
            *args: Additional args to be passed to the func.
            **kwargs: Additional Kwargs to be passed to the func.

        Returns:
            R: an object with the original return type of the function
        """
        ...

    @staticmethod
    def extract_args_and_kwargs(*args, **kwargs) -> ExtractedKwargs:
        """Takes a mix of args and kwargs for both the generative slot and the original function and extracts them.

        Returns:
            ExtractedKwargs: a dataclass of the required args for mellea and the original function.
            Either session or (backend, context) will be non-None.

        Raises:
            TODO: JAL
        """
        # Possible args for the generative slot.
        extracted = ExtractedKwargs()

        # Args can only have Mellea args. If the Mellea args get more complicated /
        # have duplicate types, use list indices rather than a match statement.
        for arg in args:
            match arg:
                case MelleaSession():
                    extracted.session = arg
                case Context():
                    extracted.context = arg
                case Backend():
                    extracted.backend = arg
                case dict():
                    extracted.model_options = arg
                case SamplingStrategy():
                    extracted.strategy = arg
                case list():
                    extracted.requirements = arg

        # TODO: JAL; make sure model opts doesn't conflict with f_kwargs here...

        T = TypeVar("T")

        def _get_val_or_err(name: str, var: T | None, new_val: T) -> T:
            """Returns the new_value if the original var is None, else raises a ValueError."""
            if var is None:
                return new_val
            else:
                raise ValueError(
                    f"passed in multiple values of {name} to generative slot: {var}, {new_val}"
                )

        # Kwargs can contain
        #   - some / all of the Mellea args
        #   - all of the function args (P.kwargs); the syntax prevents passing a P.arg to the genslot
        for key, val in kwargs.items():
            match key:
                case "m":
                    extracted.session = _get_val_or_err("m", extracted.session, val)
                case "context":
                    extracted.context = _get_val_or_err(
                        "context", extracted.context, val
                    )
                case "backend":
                    extracted.backend = _get_val_or_err(
                        "backend", extracted.backend, val
                    )
                case "model_options":
                    extracted.model_options = _get_val_or_err(
                        "model_options", extracted.model_options, val
                    )
                case "strategy":
                    extracted.strategy = _get_val_or_err(
                        "strategy", extracted.strategy, val
                    )
                case "requirements":
                    extracted.requirements = _get_val_or_err(
                        "requirements", extracted.requirements, val
                    )
                case _:
                    extracted.f_kwargs[key] = val

        # Need to check that either session is set or both backend and context are set;
        # model_options can be None.
        if extracted.session is None and (
            extracted.backend is None or extracted.context is None
        ):
            raise ValueError(
                f"need to pass in a session or a (backend and context) to generative slot; got session({extracted.session}), backend({extracted.backend}), context({extracted.context})"
            )

        return extracted

    def parts(self):
        """Not implemented."""
        raise NotImplementedError

    def format_for_llm(self) -> TemplateRepresentation:
        """Formats the instruction for Formatter use."""
        return TemplateRepresentation(
            obj=self,
            args={
                "function": self._function._function_dict,
                "arguments": [a._argument_dict for a in self._arguments],
            },
            tools=None,
            template_order=["*", "GenerativeSlot"],
        )


class SyncGenerativeSlot(GenerativeSlot, Generic[P, R]):
    @overload
    def __call__(
        self,
        context: Context,
        backend: Backend,
        requirements: list[Requirement | str] | None = None,
        strategy: SamplingStrategy | None = None,
        model_options: dict | None = None,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> tuple[R, Context]: ...

    @overload
    def __call__(
        self,
        m: MelleaSession,
        requirements: list[Requirement | str] | None = None,
        strategy: SamplingStrategy | None = None,
        model_options: dict | None = None,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R: ...

    def __call__(self, *args, **kwargs) -> tuple[R, Context] | R:
        """Call the generative slot.

        Args:
            m: MelleaSession: A mellea session (optional; must set context and backend if None)
            context: the Context object (optional; session must be set if None)
            backend: the backend used for generation (optional: session must be set if None)
            requirements: A list of requirements that the genslot output can be validated against.
            strategy: A SamplingStrategy that describes the strategy for validating and repairing/retrying. None means that no particular sampling strategy is used.
            model_options: Model options to pass to the backend.
            *args: Additional args to be passed to the func.
            **kwargs: Additional Kwargs to be passed to the func.

        Returns:
            R: an object with the original return type of the function
        """
        extracted = self.extract_args_and_kwargs(*args, **kwargs)

        slot_copy = deepcopy(self)
        arguments = bind_function_arguments(self._function._func, **extracted.f_kwargs)
        if arguments:
            for key, val in arguments.items():
                annotation = get_annotation(slot_copy._function._func, key, val)
                slot_copy._arguments.append(Argument(annotation, key, val))

        response_model = create_response_format(self._function._func)

        response, context = None, None
        if extracted.session is not None:
            response = extracted.session.act(
                slot_copy,
                requirements=extracted.requirements,
                strategy=extracted.strategy,
                format=response_model,
                model_options=extracted.model_options,
            )
        else:
            # We know these aren't None from the `extract_args_and_kwargs` function.
            assert extracted.context is not None
            assert extracted.backend is not None
            response, context = mfuncs.act(
                slot_copy,
                extracted.context,
                extracted.backend,
                requirements=extracted.requirements,
                strategy=extracted.strategy,
                format=response_model,
                model_options=extracted.model_options,
            )

        function_response: FunctionResponse[R] = response_model.model_validate_json(
            response.value  # type: ignore
        )

        if context is None:
            return function_response.result
        else:
            return function_response.result, context


class AsyncGenerativeSlot(GenerativeSlot, Generic[P, R]):
    """A generative slot component that generates asynchronously and returns a coroutine."""

    @overload
    def __call__(
        self,
        context: Context,
        backend: Backend,
        requirements: list[Requirement | str] | None = None,
        strategy: SamplingStrategy | None = None,
        model_options: dict | None = None,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Coroutine[Any, Any, tuple[R, Context]]: ...

    @overload
    def __call__(
        self,
        m: MelleaSession,
        requirements: list[Requirement | str] | None = None,
        strategy: SamplingStrategy | None = None,
        model_options: dict | None = None,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Coroutine[Any, Any, R]: ...

    def __call__(self, *args, **kwargs) -> Coroutine[Any, Any, tuple[R, Context] | R]:
        """Call the async generative slot.

        Args:
            m: MelleaSession: A mellea session (optional; must set context and backend if None)
            context: the Context object (optional; session must be set if None)
            backend: the backend used for generation (optional: session must be set if None)
            requirements: A list of requirements that the genslot output can be validated against.
            strategy: A SamplingStrategy that describes the strategy for validating and repairing/retrying. None means that no particular sampling strategy is used.
            model_options: Model options to pass to the backend.
            *args: Additional args to be passed to the func.
            **kwargs: Additional Kwargs to be passed to the func.

        Returns:
            Coroutine[Any, Any, R]: a coroutine that returns an object with the original return type of the function
        """
        extracted = self.extract_args_and_kwargs(*args, **kwargs)

        slot_copy = deepcopy(self)
        # TODO: JAL; need to figure out where / how reqs work; if we want to keep as a part of the object,
        # apply them here after the copy has happened...
        # need to change the template; add to docstring using postconditions:
        #     Postconditions:
        # - The input 'data' list will be sorted in ascending order.
        arguments = bind_function_arguments(self._function._func, **extracted.f_kwargs)
        if arguments:
            for key, val in arguments.items():
                annotation = get_annotation(slot_copy._function._func, key, val)
                slot_copy._arguments.append(Argument(annotation, key, val))

        response_model = create_response_format(self._function._func)

        # AsyncGenerativeSlots are used with async functions. In order to support that behavior,
        # they must return a coroutine object.
        async def __async_call__() -> tuple[R, Context] | R:
            response, context = None, None

            # Use the async act func so that control flow doesn't get stuck here in async event loops.
            if extracted.session is not None:
                response = await extracted.session.aact(
                    slot_copy,
                    requirements=extracted.requirements,
                    strategy=extracted.strategy,
                    format=response_model,
                    model_options=extracted.model_options,
                )
            else:
                # We know these aren't None from the `extract_args_and_kwargs` function.
                assert extracted.context is not None
                assert extracted.backend is not None
                response, context = await mfuncs.aact(
                    slot_copy,
                    extracted.context,
                    extracted.backend,
                    requirements=extracted.requirements,
                    strategy=extracted.strategy,
                    format=response_model,
                    model_options=extracted.model_options,
                )

            function_response: FunctionResponse[R] = response_model.model_validate_json(
                response.value  # type: ignore
            )

            if context is None:
                return function_response.result
            else:
                return function_response.result, context

        return __async_call__()


@overload
def generative(func: Callable[P, Awaitable[R]]) -> AsyncGenerativeSlot[P, R]: ...  # type: ignore


@overload
def generative(func: Callable[P, R]) -> SyncGenerativeSlot[P, R]: ...


# TODO: JAL Investigate changing genslots to functions and see if it fixes the defaults being populated.
def generative(func: Callable[P, R]) -> GenerativeSlot[P, R]:
    """Convert a function into an AI-powered function.

    This decorator transforms a regular Python function into one that uses an LLM
    to generate outputs. The function's entire signature - including its name,
    parameters, docstring, and type hints - is used to instruct the LLM to imitate
    that function's behavior. The output is guaranteed to match the return type
    annotation using structured outputs and automatic validation.

    Note: Works with async functions as well.

    Tip: Write the function and docstring in the most Pythonic way possible, not
    like a prompt. This ensures the function is well-documented, easily understood,
    and familiar to any Python developer. The more natural and conventional your
    function definition, the better the AI will understand and imitate it.

    Args:
        func: Function with docstring and type hints. Implementation can be empty (...).

    Returns:
        An AI-powered function that generates responses using an LLM based on the
        original function's signature and docstring.

    Raises:
        ValidationError: if the generated output cannot be parsed into the expected return type. Typically happens when the token limit for the generated output results in invalid json.

    Examples:
        >>> from mellea import generative, start_session
        >>> session = start_session()
        >>> @generative
        ... def summarize_text(text: str, max_words: int = 50) -> str:
        ...     '''Generate a concise summary of the input text.'''
        ...     ...
        >>>
        >>> summary = summarize_text(session, "Long text...", max_words=30)

        >>> from typing import List
        >>> from dataclasses import dataclass
        >>>
        >>> @dataclass
        ... class Task:
        ...     title: str
        ...     priority: str
        ...     estimated_hours: float
        >>>
        >>> @generative
        ... async def create_project_tasks(project_desc: str, count: int) -> List[Task]:
        ...     '''Generate a list of realistic tasks for a project.
        ...
        ...     Args:
        ...         project_desc: Description of the project
        ...         count: Number of tasks to generate
        ...
        ...     Returns:
        ...         List of tasks with titles, priorities, and time estimates
        ...     '''
        ...     ...
        >>>
        >>> tasks = await create_project_tasks(session, "Build a web app", 5)

        >>> @generative
        ... def analyze_code_quality(code: str) -> Dict[str, Any]:
        ...     '''Analyze code quality and provide recommendations.
        ...
        ...     Args:
        ...         code: Source code to analyze
        ...
        ...     Returns:
        ...         Dictionary containing:
        ...         - score: Overall quality score (0-100)
        ...         - issues: List of identified problems
        ...         - suggestions: List of improvement recommendations
        ...         - complexity: Estimated complexity level
        ...     '''
        ...     ...
        >>>
        >>> analysis = analyze_code_quality(
        ...     session,
        ...     "def factorial(n): return n * factorial(n-1)",
        ...     model_options={"temperature": 0.3}
        ... )

        >>> @dataclass
        ... class Thought:
        ...     title: str
        ...     body: str
        >>>
        >>> @generative
        ... def generate_chain_of_thought(problem: str, steps: int = 5) -> List[Thought]:
        ...     '''Generate a step-by-step chain of thought for solving a problem.
        ...
        ...     Args:
        ...         problem: The problem to solve or question to answer
        ...         steps: Maximum number of reasoning steps
        ...
        ...     Returns:
        ...         List of reasoning steps, each with a title and detailed body
        ...     '''
        ...     ...
        >>>
        >>> reasoning = generate_chain_of_thought(session, "How to optimize a slow database query?")
    """
    # Grab and remove the func if it exists in kwargs. Otherwise, it's the only arg.
    if inspect.iscoroutinefunction(func):
        return AsyncGenerativeSlot(func)
    else:
        return SyncGenerativeSlot(func)


# Export the decorator as the interface
__all__ = ["generative"]
