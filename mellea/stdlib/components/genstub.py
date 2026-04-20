"""A method to generate outputs based on python functions and a Generative Stub function."""

import abc
import functools
import inspect
from collections.abc import Awaitable, Callable, Coroutine
from copy import deepcopy
from dataclasses import dataclass, fields
from typing import Any, Generic, ParamSpec, TypedDict, TypeVar, get_type_hints, overload

from pydantic import BaseModel, Field, create_model

import mellea.stdlib.functional as mfuncs

from ...core import (
    Backend,
    CBlock,
    Component,
    Context,
    MelleaLogger,
    ModelOutputThunk,
    Requirement,
    SamplingStrategy,
    TemplateRepresentation,
    ValidationResult,
)
from ..requirements.requirement import reqify
from ..session import MelleaSession

P = ParamSpec("P")
R = TypeVar("R")


class FunctionResponse(BaseModel, Generic[R]):
    """Generic base class for function response formats.

    Attributes:
        result (R): The value returned by the generative function.
    """

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
    """Return Type for a Function Component.

    Attributes:
        name (str): The function's ``__name__``.
        signature (str): The function's parameter signature as a string.
        docstring (str | None): The function's docstring, or ``None`` if absent.
    """

    name: str
    signature: str
    docstring: str | None


class ArgumentDict(TypedDict):
    """Return Type for an Argument Component.

    Attributes:
        name (str | None): The parameter name.
        annotation (str | None): The parameter's type annotation as a string.
        value (str | None): The bound value for this parameter as a string.
    """

    name: str | None
    annotation: str | None
    value: str | None


class Argument:
    """A single function argument with its name, type annotation, and value.

    Args:
        annotation (str | None): The parameter's type annotation as a string.
        name (str | None): The parameter name.
        value (str | None): The bound value for this parameter as a string.
    """

    def __init__(
        self,
        annotation: str | None = None,
        name: str | None = None,
        value: str | None = None,
    ):
        """Initialize Argument with optional name, type annotation, and bound value."""
        self._argument_dict: ArgumentDict = {
            "name": name,
            "annotation": annotation,
            "value": value,
        }


class Arguments(CBlock):
    """A ``CBlock`` that renders a list of ``Argument`` objects as human-readable text.

    Each argument is formatted as ``"- name: value  (type: annotation)"`` and the
    items are newline-joined into a single string suitable for inclusion in a prompt.

    Args:
        arguments (list[Argument]): The list of bound function arguments to render.
    """

    def __init__(self, arguments: list[Argument]):
        """Initialize Arguments by rendering a list of Argument objects as a formatted string."""
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
    """Specific requirement with template for validating precondition requirements against a set of args.

    Args:
        req (Requirement): The underlying requirement to wrap. All method calls
            are delegated to this requirement.

    """

    def __init__(self, req: Requirement):
        """Initialize ArgPreconditionRequirement by wrapping an existing Requirement."""
        self.req = req

    def __getattr__(self, name):
        return getattr(self.req, name)

    def __copy__(self):
        return ArgPreconditionRequirement(req=self.req)

    def __deepcopy__(self, memo):
        return ArgPreconditionRequirement(deepcopy(self.req, memo))


class PreconditionException(Exception):
    """Exception raised when validation fails for a generative stub's arguments.

    Args:
        message (str): Human-readable description of the failure.
        validation_results (list[ValidationResult]): The individual validation
            results from the failed precondition checks.

    Attributes:
        validation (list[ValidationResult]): The validation results from the
            failed precondition checks.
    """

    def __init__(
        self, message: str, validation_results: list[ValidationResult]
    ) -> None:
        """Initialize PreconditionException with a message and the list of failed validation results."""
        super().__init__(message)
        self.validation = validation_results


class Function(Generic[P, R]):
    """Wraps a callable with its introspected ``FunctionDict`` metadata.

    Stores the original callable alongside its name, signature, and docstring
    as produced by ``describe_function``, so generative stubs can render them
    into prompts without re-inspecting the function each time.

    Args:
        func (Callable): The callable to wrap and introspect.
    """

    def __init__(self, func: Callable[P, R]):
        """Initialize Function by wrapping a callable and capturing its metadata."""
        self._func: Callable[P, R] = func
        self._function_dict: FunctionDict = describe_function(func)


def describe_function(func: Callable) -> FunctionDict:
    """Generates a FunctionDict given a function.

    Args:
        func : Callable function that needs to be passed to generative stub.

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

    Raises:
        TypeError: If required parameters from the original function are missing
            from the provided arguments.
    """
    signature = inspect.signature(func)
    try:
        bound_arguments = signature.bind(*args, **kwargs)
    except TypeError as e:
        # Provide a clear error message when parameters from the original function are missing
        if "missing" in str(e) and "required" in str(e):
            raise TypeError(
                f"generative stub is missing required parameter(s) from the original function '{func.__name__}': {e}"
            ) from e

        # Else re-raise the error if it's not the expected error.
        raise e
    bound_arguments.apply_defaults()
    return dict(bound_arguments.arguments)


@dataclass
class ExtractedArgs:
    """Used to extract the mellea args and original function args. See @generative decorator for additional notes on these fields.

    These args must match those allowed by any overload of GenerativeStub.__call__.

    Attributes:
        f_args (tuple[Any, ...]): Positional args from the original function call;
            used to detect incorrectly passed args to generative stubs.
        f_kwargs (dict[str, Any]): Keyword args intended for the original function.
        m (MelleaSession | None): The active Mellea session, if provided.
    """

    f_args: tuple[Any, ...]
    """*args from the original function, used to detect incorrectly passed args to generative stubs"""

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


class GenerativeStub(Component[R], Generic[P, R]):
    """Abstract base class for AI-powered function wrappers produced by ``@generative``.

    A ``GenerativeStub`` wraps a callable and uses an LLM to generate its output.
    Subclasses (``SyncGenerativeStub``, ``AsyncGenerativeStub``) implement
    ``__call__`` for synchronous and asynchronous invocation respectively.
    The function's signature, docstring, and type hints are rendered into a prompt
    so the LLM can imitate the function's intended behaviour.

    Args:
        func (Callable): The function whose behaviour the LLM should imitate.

    Attributes:
        precondition_requirements (list[Requirement]): Requirements validated
            against the function's input arguments before generation.
        requirements (list[Requirement]): Requirements validated against the
            LLM's generated output.
    """

    def __init__(self, func: Callable[P, R]):
        """Initialize GenerativeStub by wrapping the given callable and validating its parameter names.

        Raises:
            ValueError: if the decorated function has a parameter name used by generative stubs
        """
        sig = inspect.signature(func)
        problematic_param_names: list[str] = []
        for param in sig.parameters.keys():
            if param in _disallowed_param_names:
                problematic_param_names.append(param)

        if len(problematic_param_names):
            raise ValueError(
                f"cannot create a generative stub with disallowed parameter names: {problematic_param_names}"
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
        """Call the generative stub. See subclasses for more information."""
        ...

    @staticmethod
    def extract_args_and_kwargs(*args, **kwargs) -> ExtractedArgs:
        """Take a mix of args and kwargs for both the generative stub and the original function and extract them. Ensures the original function's args are all kwargs.

        Args:
            args: Positional arguments; the first must be either a
                ``MelleaSession`` or a ``Context`` instance.
            kwargs: Keyword arguments for both the generative stub machinery
                (e.g. ``m``, ``context``, ``backend``, ``requirements``) and the
                wrapped function's own parameters.

        Returns:
            ExtractedArgs: A dataclass of the required args for mellea and the
            original function. Either session or (backend, context) will be
            non-None.

        Raises:
            TypeError: If any of the original function's parameters were passed
                as positional args or if required mellea parameters are missing.
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
                    "generative stub requires either a MelleaSession (m=...) or both a Context and Backend (context=..., backend=...) to be provided as the first argument(s)"
                ) from e

            # If it's not the expected err, simply re-raise it.
            raise e

        if len(extracted.f_args) > 0:
            raise TypeError(
                "generative stubs do not accept positional args from the decorated function; use keyword args instead"
            )

        return extracted

    def parts(self) -> list[Component | CBlock]:
        """Return the constituent parts of this generative stub component.

        Includes the rendered arguments block (if arguments have been bound)
        and any requirements attached to this stub.

        Returns:
            list[Component | CBlock]: List of argument blocks and requirements.
        """
        cs: list = []
        if self._arguments is not None:
            cs.append(self._arguments)
        cs.extend(self.requirements)
        return cs

    def format_for_llm(self) -> TemplateRepresentation:
        """Format this generative stub for the language model.

        Builds a ``TemplateRepresentation`` containing the function metadata
        (name, signature, docstring), the bound arguments, and any requirement
        descriptions.

        Returns:
            TemplateRepresentation: The formatted representation ready for the
            ``Formatter`` to render into a prompt.
        """
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
            template_order=["*", "GenerativeStub"],
        )

    def _parse(self, computed: ModelOutputThunk) -> R:
        """Parse the model output. Returns the original function's return type."""
        function_response: FunctionResponse[R] = (
            self._response_model.model_validate_json(
                computed.value  # type: ignore
            )
        )

        return function_response.result


class SyncGenerativeStub(GenerativeStub, Generic[P, R]):
    """A synchronous generative stub that blocks until the LLM response is ready.

    Returned by ``@generative`` when the decorated function is not a coroutine.
    ``__call__`` returns the parsed result directly (when a session is passed) or a
    ``(result, context)`` tuple (when a context and backend are passed).
    """

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
        """Call the generative stub.

        Args:
            m: MelleaSession: A mellea session (optional: must set context and backend if None)
            context: the Context object (optional: session must be set if None)
            backend: the backend used for generation (optional: session must be set if None)
            precondition_requirements: A list of requirements that the genstub inputs are validated against; does not use a sampling strategy.
            requirements: A list of requirements that the genstub output can be validated against.
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

        stub_copy = deepcopy(self)
        if extracted.requirements is not None:
            stub_copy.requirements = [reqify(r) for r in extracted.requirements]

        if extracted.precondition_requirements is not None:
            stub_copy.precondition_requirements = [
                ArgPreconditionRequirement(reqify(r))
                for r in extracted.precondition_requirements
            ]

        arguments = bind_function_arguments(self._function._func, **extracted.f_kwargs)
        if arguments:
            stub_args: list[Argument] = []
            for key, val in arguments.items():
                stub_args.append(get_argument(stub_copy._function._func, key, val))
            stub_copy._arguments = Arguments(stub_args)

        # Do precondition validation first.
        if stub_copy._arguments is not None:
            if extracted.m is not None:
                val_results = extracted.m.validate(
                    reqs=stub_copy.precondition_requirements,
                    model_options=extracted.model_options,
                    output=ModelOutputThunk(stub_copy._arguments.value),
                )
            else:
                # We know these aren't None from the `extract_args_and_kwargs` function.
                assert extracted.context is not None
                assert extracted.backend is not None
                val_results = mfuncs.validate(
                    reqs=stub_copy.precondition_requirements,
                    context=extracted.context,
                    backend=extracted.backend,
                    model_options=extracted.model_options,
                    output=ModelOutputThunk(stub_copy._arguments.value),
                )

            # No retries if precondition validation fails.
            if not all(bool(val_result) for val_result in val_results):
                MelleaLogger.get_logger().error(
                    "generative stub arguments did not satisfy precondition requirements"
                )
                raise PreconditionException(
                    "generative stub arguments did not satisfy precondition requirements",
                    validation_results=val_results,
                )

        elif len(stub_copy.precondition_requirements) > 0:
            MelleaLogger.get_logger().warning(
                "calling a generative stub with precondition requirements but no args to validate the preconditions against; ignoring precondition validation"
            )

        response, context = None, None
        if extracted.m is not None:
            response = extracted.m.act(
                stub_copy,
                requirements=stub_copy.requirements,
                strategy=extracted.strategy,
                format=self._response_model,
                model_options=extracted.model_options,
            )
        else:
            # We know these aren't None from the `extract_args_and_kwargs` function.
            assert extracted.context is not None
            assert extracted.backend is not None
            response, context = mfuncs.act(
                stub_copy,
                extracted.context,
                extracted.backend,
                requirements=stub_copy.requirements,
                strategy=extracted.strategy,
                format=self._response_model,
                model_options=extracted.model_options,
            )

        assert response.parsed_repr is not None
        if context is None:
            return response.parsed_repr
        else:
            return response.parsed_repr, context


class AsyncGenerativeStub(GenerativeStub, Generic[P, R]):
    """A generative stub component that generates asynchronously and returns a coroutine."""

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
        """Call the async generative stub.

        Args:
            m: MelleaSession: A mellea session (optional: must set context and backend if None)
            context: the Context object (optional: session must be set if None)
            backend: the backend used for generation (optional: session must be set if None)
            precondition_requirements: A list of requirements that the genstub inputs are validated against; does not use a sampling strategy.
            requirements: A list of requirements that the genstub output can be validated against.
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

        stub_copy = deepcopy(self)
        if extracted.requirements is not None:
            stub_copy.requirements = [reqify(r) for r in extracted.requirements]

        if extracted.precondition_requirements is not None:
            stub_copy.precondition_requirements = [
                ArgPreconditionRequirement(reqify(r))
                for r in extracted.precondition_requirements
            ]

        arguments = bind_function_arguments(self._function._func, **extracted.f_kwargs)
        if arguments:
            stub_args: list[Argument] = []
            for key, val in arguments.items():
                stub_args.append(get_argument(stub_copy._function._func, key, val))
            stub_copy._arguments = Arguments(stub_args)

        # AsyncGenerativeStubs are used with async functions. In order to support that behavior,
        # they must return a coroutine object.
        async def __async_call__() -> tuple[R, Context] | R:
            """Use async calls so that control flow doesn't get stuck here in async event loops."""
            response, context = None, None

            # Do precondition validation first.
            if stub_copy._arguments is not None:
                if extracted.m is not None:
                    val_results = await extracted.m.avalidate(
                        reqs=stub_copy.precondition_requirements,
                        model_options=extracted.model_options,
                        output=ModelOutputThunk(stub_copy._arguments.value),
                    )
                else:
                    # We know these aren't None from the `extract_args_and_kwargs` function.
                    assert extracted.context is not None
                    assert extracted.backend is not None
                    val_results = await mfuncs.avalidate(
                        reqs=stub_copy.precondition_requirements,
                        context=extracted.context,
                        backend=extracted.backend,
                        model_options=extracted.model_options,
                        output=ModelOutputThunk(stub_copy._arguments.value),
                    )

                # No retries if precondition validation fails.
                if not all(bool(val_result) for val_result in val_results):
                    MelleaLogger.get_logger().error(
                        "generative stub arguments did not satisfy precondition requirements"
                    )
                    raise PreconditionException(
                        "generative stub arguments did not satisfy precondition requirements",
                        validation_results=val_results,
                    )

            elif len(stub_copy.precondition_requirements) > 0:
                MelleaLogger.get_logger().warning(
                    "calling a generative stub with precondition requirements but no args to validate the preconditions against; ignoring precondition validation"
                )

            if extracted.m is not None:
                response = await extracted.m.aact(
                    stub_copy,
                    requirements=stub_copy.requirements,
                    strategy=extracted.strategy,
                    format=self._response_model,
                    model_options=extracted.model_options,
                    await_result=True,
                )
            else:
                # We know these aren't None from the `extract_args_and_kwargs` function.
                assert extracted.context is not None
                assert extracted.backend is not None
                response, context = await mfuncs.aact(
                    stub_copy,
                    extracted.context,
                    extracted.backend,
                    requirements=stub_copy.requirements,
                    strategy=extracted.strategy,
                    format=self._response_model,
                    model_options=extracted.model_options,
                    await_result=True,
                )

            assert response.is_computed(), (
                "unexpectedly received uncomputed model output thunk in async generative stub"
            )
            assert response.parsed_repr is not None
            if context is None:
                return response.parsed_repr
            else:
                return response.parsed_repr, context

        return __async_call__()


@overload
def generative(func: Callable[P, Awaitable[R]]) -> AsyncGenerativeStub[P, R]: ...  # type: ignore


@overload
def generative(func: Callable[P, R]) -> SyncGenerativeStub[P, R]: ...


def generative(func: Callable[P, R]) -> GenerativeStub[P, R]:
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
        *precondition_requirements*: list[Requirements | str] | None: A list of requirements that the genstub inputs are validated against; raises an err if not met.
        *requirements*: list[Requirement | str] | None: A list of requirements that the genstub output can be validated against.
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
        ValueError: (raised by @generative) if the decorated function has a parameter name used by generative stubs
        ValidationError: (raised when calling the generative stub) if the generated output cannot be parsed into the expected return type. Typically happens when the token limit for the generated output results in invalid json.
        TypeError: (raised when calling the generative stub) if any of the original function's parameters were passed as positional args
        PreconditionException: (raised when calling the generative stub) if the precondition validation of the args fails; catch the exception to get the validation results

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
        return AsyncGenerativeStub(func)
    else:
        return SyncGenerativeStub(func)


# Export the decorator as the interface. Export the specific exception for debugging.
__all__ = ["PreconditionException", "generative"]
