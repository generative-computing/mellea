"""A method to generate outputs based on python functions and a Generative Slot function."""

import asyncio
import functools
import inspect
from collections.abc import Callable, Coroutine
from copy import deepcopy
from typing import Any, Generic, ParamSpec, TypedDict, TypeVar, get_type_hints

from pydantic import BaseModel, Field, create_model

from mellea.stdlib.base import Component, TemplateRepresentation
from mellea.stdlib.session import MelleaSession, get_session

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

    def __call__(
        self,
        m: MelleaSession | None = None,
        model_options: dict | None = None,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        """Call the generative slot.

        Args:
            m: MelleaSession: A mellea session (optional, uses context if None)
            model_options: Model options to pass to the backend.
            *args: Additional args to be passed to the func.
            **kwargs: Additional Kwargs to be passed to the func.

        Returns:
            R: an object with the original return type of the function
        """
        if m is None:
            m = get_session()
        slot_copy = deepcopy(self)
        arguments = bind_function_arguments(self._function._func, *args, **kwargs)
        if arguments:
            for key, val in arguments.items():
                annotation = get_annotation(slot_copy._function._func, key, val)
                slot_copy._arguments.append(Argument(annotation, key, val))

        response_model = create_response_format(self._function._func)

        response = m.act(slot_copy, format=response_model, model_options=model_options)

        function_response: FunctionResponse[R] = response_model.model_validate_json(
            response.value  # type: ignore
        )

        return function_response.result

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


class AsyncGenerativeSlot(GenerativeSlot, Generic[P, R]):
    """A generative slot component that generates asynchronously and returns a coroutine."""

    def __call__(
        self,
        m: MelleaSession | None = None,
        model_options: dict | None = None,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Coroutine[Any, Any, R]:
        """Call the async generative slot.

        Args:
            m: MelleaSession: A mellea session (optional, uses context if None)
            **kwargs: Additional Kwargs to be passed to the func

        Returns:
            Coroutine[Any, Any, R]: a coroutine that returns an object with the original return type of the function
        """
        if m is None:
            m = get_session()
        slot_copy = deepcopy(self)
        arguments = bind_function_arguments(self._function._func, *args, **kwargs)
        if arguments:
            for key, val in arguments.items():
                annotation = get_annotation(slot_copy._function._func, key, val)
                slot_copy._arguments.append(Argument(annotation, key, val))

        response_model = create_response_format(self._function._func)

        # AsyncGenerativeSlots are used with async functions. In order to support that behavior,
        # they must return a coroutine object.
        async def __async_call__():
            # Use the async act func so that control flow doesn't get stuck here in async event loops.
            response = await m.aact(
                slot_copy, format=response_model, model_options=model_options
            )

            function_response: FunctionResponse[R] = response_model.model_validate_json(
                response.value  # type: ignore
            )
            return function_response.result

        return __async_call__()


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
    if inspect.iscoroutinefunction(func):
        return AsyncGenerativeSlot(func)
    else:
        return GenerativeSlot(func)


# Export the decorator as the interface
__all__ = ["generative"]


if __name__ == "__main__":
    from mellea import start_session

    with start_session():

        async def asyncly() -> int: ...

        out = asyncly()

        @generative
        async def test_async(num: int) -> bool: ...

        @generative
        def test_sync(truthy: bool) -> int: ...

        print("running sync")
        print(test_sync(m=None, model_options=None, truthy=False))

        async def runmany():
            print(await test_async(m=None, model_options=None, num=6))
            print(await test_async(m=None, model_options=None, num=4))
            print(await test_async(m=None, model_options=None, num=5))

            coros = [
                test_async(m=None, model_options=None, num=1),
                test_async(m=None, model_options=None, num=2),
                test_async(m=None, model_options=None, num=3),
            ]
            results = await asyncio.gather(*coros)
            print(results)

        print("running async")
        asyncio.run(runmany())
