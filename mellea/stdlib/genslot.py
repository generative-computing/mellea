"""A method to generate outputs based on python functions and a Generative Slot function."""

import abc
import functools
import inspect
from collections.abc import Awaitable, Callable, Coroutine
from copy import copy, deepcopy
from dataclasses import dataclass, fields
from typing import Any, Generic, ParamSpec, TypedDict, TypeVar, get_type_hints, overload

from pydantic import BaseModel, Field, create_model

import mellea.stdlib.functional as mfuncs
from mellea.backends import Backend
from mellea.helpers.fancy_logger import FancyLogger
from mellea.stdlib.base import (
    CBlock,
    Component,
    Context,
    ModelOutputThunk,
    TemplateRepresentation,
)
from mellea.stdlib.requirement import Requirement, ValidationResult, reqify
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


class Argument:
    """An Argument."""

    def __init__(
        self,
        annotation: str | None = None,
        name: str | None = None,
        value: str | None = None,
    ):
        """An Argument."""
        self._argument_dict: ArgumentDict = {
            "name": name,
            "annotation": annotation,
            "value": value,
        }


class Arguments(CBlock):
    def __init__(self, arguments: list[Argument]):
        """Create a textual representation of a list of arguments."""
        # Make meta the original list of arguments and create a list of textual representations.
        meta: dict[str, Any] = {}
        text_args = []
        for arg in arguments:
            assert arg._argument_dict["name"] is not None
            meta[arg._argument_dict["name"]] = arg
            text_args.append(
                f"- {arg._argument_dict['name']}: {arg._argument_dict['value']}  (type: {arg._argument_dict['annotation']})"
            )

        super().__init__("\n".join(text_args), meta)


class ArgPreconditionRequirement(Requirement):
    """Specific requirement with template for validating precondition requirements against a set of args."""

    def __init__(self, req: Requirement):
        """Can only be instantiated from existing requirements. All function calls are delegated to the underlying requirement."""
        self.req = req

    def __getattr__(self, name):
        return getattr(self.req, name)

    def __copy__(self):
        return ArgPreconditionRequirement(req=self.req)

    def __deepcopy__(self, memo):
        return ArgPreconditionRequirement(deepcopy(self.req, memo))


class PreconditionException(Exception):
    """Exception raised when validation fails for a generative slot's arguments."""

    def __init__(
        self, message: str, validation_results: list[ValidationResult]
    ) -> None:
        """Exception raised when validation fails for a generative slot's arguments.

        Args:
            message: the error message
            validation_results: the list of validation results from the failed preconditions
        """
        super().__init__(message)
        self.validation = validation_results


class Function(Generic[P, R]):
    """A Function."""

    def __init__(self, func: Callable[P, R]):
        """A Function."""
        self._func: Callable[P, R] = func
        self._function_dict: FunctionDict = describe_function(func)


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


def get_argument(func: Callable, key: str, val: Any) -> Argument:
    """Returns an argument given a parameter.

    Note: Performs additional formatting for string objects, putting them in quotes.

    Args:
        func : Callable Function
        key : Arg key
        val : Arg value

    Returns:
        Argument: an argument object representing the given parameter.
    """
    sig = inspect.signature(func)
    param = sig.parameters.get(key)
    if param and param.annotation is not inspect.Parameter.empty:
        param_type = param.annotation
    else:
        param_type = type(val)

    if param_type is str:
        val = f'"{val!s}"'

    return Argument(str(param_type), key, val)


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
    try:
        bound_arguments = signature.bind(*args, **kwargs)
    except TypeError as e:
        # Provide a clear error message when parameters from the original function are missing
        if "missing" in str(e) and "required" in str(e):
            raise TypeError(
                f"generative slot is missing required parameter(s) from the original function '{func.__name__}': {e}"
            ) from e

        # Else re-raise the error if it's not the expected error.
        raise e
    bound_arguments.apply_defaults()
    return dict(bound_arguments.arguments)


@dataclass
class ExtractedArgs:
    """Used to extract the mellea args and original function args. See @generative decorator for additional notes on these fields.

    These args must match those allowed by any overload of GenerativeSlot.__call__.
    """

    f_args: tuple[Any, ...]
    """*args from the original function, used to detect incorrectly passed args to generative slots"""

    f_kwargs: dict[str, Any]
    """**kwargs from the original function"""

    m: MelleaSession | None = None
    context: Context | None = None
    backend: Backend | None = None
    model_options: dict | None = None
    strategy: SamplingStrategy | None = None

    precondition_requirements: list[Requirement | str] | None = None
    """requirements used to check the input"""

    requirements: list[Requirement | str] | None = None
    """requirements used to check the output"""

    def __init__(self):
        """Used to extract the mellea args and original function args."""
        self.f_args = tuple()
        self.f_kwargs = {}


_disallowed_param_names = [field.name for field in fields(ExtractedArgs())]
"""A list of parameter names used by Mellea. Cannot use these in functions decorated with @generative."""


class GenerativeSlot(Component[R], Generic[P, R]):
    """A generative slot component."""

    def __init__(self, func: Callable[P, R]):
        """A generative slot function that converts a given `func` to a generative slot.

        Args:
            func: A callable function

        Raises:
            ValueError: if the decorated function has a parameter name used by generative slots
        """
        sig = inspect.signature(func)
        problematic_param_names: list[str] = []
        for param in sig.parameters.keys():
            if param in _disallowed_param_names:
                problematic_param_names.append(param)

        if len(problematic_param_names):
            raise ValueError(
                f"cannot create a generative slot with disallowed parameter names: {problematic_param_names}"
            )

        self._function = Function(func)
        self._arguments: Arguments | None = None
        functools.update_wrapper(self, func)

        self._response_model = create_response_format(self._function._func)

        # Set when calling the decorated func.
        self.precondition_requirements: list[Requirement] = []
        self.requirements: list[Requirement] = []

    @abc.abstractmethod
    def __call__(self, *args, **kwargs) -> tuple[R, Context] | R:
        """Call the generative slot. See subclasses for more information."""
        ...

    @staticmethod
    def extract_args_and_kwargs(*args, **kwargs) -> ExtractedArgs:
        """Takes a mix of args and kwargs for both the generative slot and the original function and extracts them. Ensures the original function's args are all kwargs.

        Returns:
            ExtractedArgs: a dataclass of the required args for mellea and the original function.
            Either session or (backend, context) will be non-None.

        Raises:
            TypeError: if any of the original function's parameters were passed as positional args
        """

        def _session_extract_args_and_kwargs(
            m: MelleaSession,
            precondition_requirements: list[Requirement | str] | None = None,
            requirements: list[Requirement | str] | None = None,
            strategy: SamplingStrategy | None = None,
            model_options: dict | None = None,
            *args,
            **kwargs,
        ):
            """Helper function for extracting args. Used when a session is passed."""
            extracted = ExtractedArgs()
            extracted.m = m
            extracted.precondition_requirements = precondition_requirements
            extracted.requirements = requirements
            extracted.strategy = strategy
            extracted.model_options = model_options
            extracted.f_args = args
            extracted.f_kwargs = kwargs
            return extracted

        def _context_backend_extract_args_and_kwargs(
            context: Context,
            backend: Backend,
            precondition_requirements: list[Requirement | str] | None = None,
            requirements: list[Requirement | str] | None = None,
            strategy: SamplingStrategy | None = None,
            model_options: dict | None = None,
            *args,
            **kwargs,
        ):
            """Helper function for extracting args. Used when a context and a backend are passed."""
            extracted = ExtractedArgs()
            extracted.context = context
            extracted.backend = backend
            extracted.precondition_requirements = precondition_requirements
            extracted.requirements = requirements
            extracted.strategy = strategy
            extracted.model_options = model_options
            extracted.f_args = args
            extracted.f_kwargs = kwargs
            return extracted

        # Determine which overload was used:
        # - if there's args, the first arg must either be a `MelleaSession` or a `Context`
        # - otherwise, just check the kwargs for a "m" that is type `MelleaSession`
        using_session_overload = False
        if len(args) > 0:
            possible_session = args[0]
        else:
            possible_session = kwargs.get("m", None)
        if isinstance(possible_session, MelleaSession):
            using_session_overload = True

        # Call the appropriate function and let python handle the arg/kwarg extraction.
        try:
            if using_session_overload:
                extracted = _session_extract_args_and_kwargs(*args, **kwargs)
            else:
                extracted = _context_backend_extract_args_and_kwargs(*args, **kwargs)
        except TypeError as e:
            # Provide a clear error message when required mellea parameters are missing
            if "missing" in str(e) and (
                "context" in str(e) or "backend" in str(e) or "m" in str(e)
            ):
                raise TypeError(
                    "generative slot requires either a MelleaSession (m=...) or both a Context and Backend (context=..., backend=...) to be provided as the first argument(s)"
                ) from e

            # If it's not the expected err, simply re-raise it.
            raise e

        if len(extracted.f_args) > 0:
            raise TypeError(
                "generative slots do not accept positional args from the decorated function; use keyword args instead"
            )

        return extracted

    def parts(self):
        """Not implemented."""
        cs: list = []
        if self._arguments is not None:
            cs.append(self._arguments)
        cs.extend(self.requirements)
        return cs

    def format_for_llm(self) -> TemplateRepresentation:
        """Formats the instruction for Formatter use."""
        return TemplateRepresentation(
            obj=self,
            args={
                "function": self._function._function_dict,
                "arguments": self._arguments,
                "requirements": [
                    r.description
                    for r in self.requirements
                    if r.description is not None
                    and r.description != ""
                    and not r.check_only
                ],  # Same conditions on requirements as in instruction.
            },
            tools=None,
            template_order=["*", "GenerativeSlot"],
        )

    def parse(self, computed: ModelOutputThunk) -> R:
        """Parse the model output. Returns the original function's return type."""
        function_response: FunctionResponse[R] = (
            self._response_model.model_validate_json(
                computed.value  # type: ignore
            )
        )

        return function_response.result


class SyncGenerativeSlot(GenerativeSlot, Generic[P, R]):
    @overload
    def __call__(
        self,
        context: Context,
        backend: Backend,
        precondition_requirements: list[Requirement | str] | None = None,
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
        precondition_requirements: list[Requirement | str] | None = None,
        requirements: list[Requirement | str] | None = None,
        strategy: SamplingStrategy | None = None,
        model_options: dict | None = None,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R: ...

    def __call__(self, *args, **kwargs) -> tuple[R, Context] | R:
        """Call the generative slot.

        Args:
            m: MelleaSession: A mellea session (optional: must set context and backend if None)
            context: the Context object (optional: session must be set if None)
            backend: the backend used for generation (optional: session must be set if None)
            precondition_requirements: A list of requirements that the genslot inputs are validated against; does not use a sampling strategy.
            requirements: A list of requirements that the genslot output can be validated against.
            strategy: A SamplingStrategy that describes the strategy for validating and repairing/retrying. None means that no particular sampling strategy is used.
            model_options: Model options to pass to the backend.
            *args: Additional args to be passed to the func.
            **kwargs: Additional Kwargs to be passed to the func.

        Returns:
            Coroutine[Any, Any, R]: a coroutine that returns an object with the original return type of the function

        Raises:
            TypeError: if any of the original function's parameters were passed as positional args
            PreconditionException: if the precondition validation fails, catch the err to get the validation results
        """
        extracted = self.extract_args_and_kwargs(*args, **kwargs)

        slot_copy = deepcopy(self)
        if extracted.requirements is not None:
            slot_copy.requirements = [reqify(r) for r in extracted.requirements]

        if extracted.precondition_requirements is not None:
            slot_copy.precondition_requirements = [
                ArgPreconditionRequirement(reqify(r))
                for r in extracted.precondition_requirements
            ]

        arguments = bind_function_arguments(self._function._func, **extracted.f_kwargs)
        if arguments:
            slot_args: list[Argument] = []
            for key, val in arguments.items():
                slot_args.append(get_argument(slot_copy._function._func, key, val))
            slot_copy._arguments = Arguments(slot_args)

        # Do precondition validation first.
        if slot_copy._arguments is not None:
            if extracted.m is not None:
                val_results = extracted.m.validate(
                    reqs=slot_copy.precondition_requirements,
                    model_options=extracted.model_options,
                    output=ModelOutputThunk(slot_copy._arguments.value),
                )
            else:
                # We know these aren't None from the `extract_args_and_kwargs` function.
                assert extracted.context is not None
                assert extracted.backend is not None
                val_results = mfuncs.validate(
                    reqs=slot_copy.precondition_requirements,
                    context=extracted.context,
                    backend=extracted.backend,
                    model_options=extracted.model_options,
                    output=ModelOutputThunk(slot_copy._arguments.value),
                )

            # No retries if precondition validation fails.
            if not all(bool(val_result) for val_result in val_results):
                FancyLogger.get_logger().error(
                    "generative slot arguments did not satisfy precondition requirements"
                )
                raise PreconditionException(
                    "generative slot arguments did not satisfy precondition requirements",
                    validation_results=val_results,
                )

        elif len(slot_copy.precondition_requirements) > 0:
            FancyLogger.get_logger().warning(
                "calling a generative slot with precondition requirements but no args to validate the preconditions against; ignoring precondition validation"
            )

        response, context = None, None
        if extracted.m is not None:
            response = extracted.m.act(
                slot_copy,
                requirements=slot_copy.requirements,
                strategy=extracted.strategy,
                format=self._response_model,
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
                requirements=slot_copy.requirements,
                strategy=extracted.strategy,
                format=self._response_model,
                model_options=extracted.model_options,
            )

        assert response.parsed_repr is not None
        if context is None:
            return response.parsed_repr
        else:
            return response.parsed_repr, context


class AsyncGenerativeSlot(GenerativeSlot, Generic[P, R]):
    """A generative slot component that generates asynchronously and returns a coroutine."""

    @overload
    def __call__(
        self,
        context: Context,
        backend: Backend,
        precondition_requirements: list[Requirement | str] | None = None,
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
        precondition_requirements: list[Requirement | str] | None = None,
        requirements: list[Requirement | str] | None = None,
        strategy: SamplingStrategy | None = None,
        model_options: dict | None = None,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Coroutine[Any, Any, R]: ...

    def __call__(self, *args, **kwargs) -> Coroutine[Any, Any, tuple[R, Context] | R]:
        """Call the async generative slot.

        Args:
            m: MelleaSession: A mellea session (optional: must set context and backend if None)
            context: the Context object (optional: session must be set if None)
            backend: the backend used for generation (optional: session must be set if None)
            precondition_requirements: A list of requirements that the genslot inputs are validated against; does not use a sampling strategy.
            requirements: A list of requirements that the genslot output can be validated against.
            strategy: A SamplingStrategy that describes the strategy for validating and repairing/retrying. None means that no particular sampling strategy is used.
            model_options: Model options to pass to the backend.
            *args: Additional args to be passed to the func.
            **kwargs: Additional Kwargs to be passed to the func.

        Returns:
            Coroutine[Any, Any, R]: a coroutine that returns an object with the original return type of the function

        Raises:
            TypeError: if any of the original function's parameters were passed as positional args
            PreconditionException: if the precondition validation fails, catch the err to get the validation results
        """
        extracted = self.extract_args_and_kwargs(*args, **kwargs)

        slot_copy = deepcopy(self)
        if extracted.requirements is not None:
            slot_copy.requirements = [reqify(r) for r in extracted.requirements]

        if extracted.precondition_requirements is not None:
            slot_copy.precondition_requirements = [
                ArgPreconditionRequirement(reqify(r))
                for r in extracted.precondition_requirements
            ]

        arguments = bind_function_arguments(self._function._func, **extracted.f_kwargs)
        if arguments:
            slot_args: list[Argument] = []
            for key, val in arguments.items():
                slot_args.append(get_argument(slot_copy._function._func, key, val))
            slot_copy._arguments = Arguments(slot_args)

        # AsyncGenerativeSlots are used with async functions. In order to support that behavior,
        # they must return a coroutine object.
        async def __async_call__() -> tuple[R, Context] | R:
            """Use async calls so that control flow doesn't get stuck here in async event loops."""
            response, context = None, None

            # Do precondition validation first.
            if slot_copy._arguments is not None:
                if extracted.m is not None:
                    val_results = await extracted.m.avalidate(
                        reqs=slot_copy.precondition_requirements,
                        model_options=extracted.model_options,
                        output=ModelOutputThunk(slot_copy._arguments.value),
                    )
                else:
                    # We know these aren't None from the `extract_args_and_kwargs` function.
                    assert extracted.context is not None
                    assert extracted.backend is not None
                    val_results = await mfuncs.avalidate(
                        reqs=slot_copy.precondition_requirements,
                        context=extracted.context,
                        backend=extracted.backend,
                        model_options=extracted.model_options,
                        output=ModelOutputThunk(slot_copy._arguments.value),
                    )

                # No retries if precondition validation fails.
                if not all(bool(val_result) for val_result in val_results):
                    FancyLogger.get_logger().error(
                        "generative slot arguments did not satisfy precondition requirements"
                    )
                    raise PreconditionException(
                        "generative slot arguments did not satisfy precondition requirements",
                        validation_results=val_results,
                    )

            elif len(slot_copy.precondition_requirements) > 0:
                FancyLogger.get_logger().warning(
                    "calling a generative slot with precondition requirements but no args to validate the preconditions against; ignoring precondition validation"
                )

            if extracted.m is not None:
                response = await extracted.m.aact(
                    slot_copy,
                    requirements=slot_copy.requirements,
                    strategy=extracted.strategy,
                    format=self._response_model,
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
                    requirements=slot_copy.requirements,
                    strategy=extracted.strategy,
                    format=self._response_model,
                    model_options=extracted.model_options,
                )

            assert response.parsed_repr is not None
            if context is None:
                return response.parsed_repr
            else:
                return response.parsed_repr, context

        return __async_call__()


@overload
def generative(func: Callable[P, Awaitable[R]]) -> AsyncGenerativeSlot[P, R]: ...  # type: ignore


@overload
def generative(func: Callable[P, R]) -> SyncGenerativeSlot[P, R]: ...


def generative(func: Callable[P, R]) -> GenerativeSlot[P, R]:
    """Convert a function into an AI-powered function.

    This decorator transforms a regular Python function into one that uses an LLM
    to generate outputs. The function's entire signature - including its name,
    parameters, docstring, and type hints - is used to instruct the LLM to imitate
    that function's behavior. The output is guaranteed to match the return type
    annotation using structured outputs and automatic validation.

    Notes:
    - Works with async functions as well.
    - Must pass all parameters for the original function as keyword args.
    - Most python type-hinters will not show the default values but will correctly infer them;
    this means that you can set default values in the decorated function and the only necessary values will be a session or a (context, backend).

    Tip: Write the function and docstring in the most Pythonic way possible, not
    like a prompt. This ensures the function is well-documented, easily understood,
    and familiar to any Python developer. The more natural and conventional your
    function definition, the better the AI will understand and imitate it.

    The new function has the following additional args:
        *m*: MelleaSession: A mellea session (optional: must set context and backend if None)
        *context*: Context: the Context object (optional: session must be set if None)
        *backend*: Backend: the backend used for generation (optional: session must be set if None)
        *precondition_requirements*: list[Requirements | str] | None: A list of requirements that the genslot inputs are validated against; raises an err if not met.
        *requirements*: list[Requirement | str] | None: A list of requirements that the genslot output can be validated against.
        *strategy*: SamplingStrategy | None: A SamplingStrategy that describes the strategy for validating and repairing/retrying. None means that no particular sampling strategy is used.
        *model_options*: dict | None: Model options to pass to the backend.

    The requirements and validation for the generative function operate over a textual representation
    of the arguments / outputs (not their python objects).

    Args:
        func: Function with docstring and type hints. Implementation can be empty (...).

    Returns:
        An AI-powered function that generates responses using an LLM based on the
        original function's signature and docstring.

    Raises:
        ValueError: (raised by @generative) if the decorated function has a parameter name used by generative slots
        ValidationError: (raised when calling the generative slot) if the generated output cannot be parsed into the expected return type. Typically happens when the token limit for the generated output results in invalid json.
        TypeError: (raised when calling the generative slot) if any of the original function's parameters were passed as positional args
        PreconditionException: (raised when calling the generative slot) if the precondition validation of the args fails; catch the exception to get the validation results

    Examples:
        ```python
        >>> from mellea import generative, start_session
        >>> session = start_session()
        >>> @generative
        ... def summarize_text(text: str, max_words: int = 50) -> str:
        ...     '''Generate a concise summary of the input text.'''
        ...     ...
        >>>
        >>> summary = summarize_text(session, text="Long text...", max_words=30)

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
        >>> tasks = await create_project_tasks(session, project_desc="Build a web app", count=5)

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
        ...     code="def factorial(n): return n * factorial(n-1)",
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
        >>> reasoning = generate_chain_of_thought(session, problem="How to optimize a slow database query?")
        ```
    """
    if inspect.iscoroutinefunction(func):
        return AsyncGenerativeSlot(func)
    else:
        return SyncGenerativeSlot(func)


# Export the decorator as the interface. Export the specific exception for debugging.
__all__ = ["PreconditionException", "generative"]
