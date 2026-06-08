"""Pre-configured sampling presets bundling requirements and strategies.

Sampling presets encapsulate common patterns for code generation validation,
bundling a list of requirements with an optimized sampling strategy and feedback
configuration. Use presets to reduce boilerplate when setting up common workflows
like Python code generation or matplotlib plotting.

Examples:
    Generate Python code with automatic repair:

        >>> from mellea.stdlib.sampling import python_code_generation_sampling
        >>> preset = python_code_generation_sampling(allowed_imports=["numpy"])
        >>> result = session.instruct(
        ...     "Write a function to compute the mean",
        ...     requirements=preset.requirements,
        ...     strategy=preset.strategy,
        ... )

    Generate matplotlib plots with stricter constraints:

        >>> preset = python_plotting_sampling(output_path="/tmp/plot.png")
        >>> result = session.instruct(
        ...     "Create a scatter plot",
        ...     requirements=preset.requirements,
        ...     strategy=preset.strategy,
        ... )
"""

from dataclasses import dataclass
from typing import Generic, Literal

from ...core import Requirement, S, SamplingStrategy
from ..requirements.python_tools import python_code_generation_requirements
from .feedback import ModelFriendlyRepairStrategy


@dataclass
class SamplingPreset(Generic[S]):
    """Bundle of requirements and strategy for a specific use case.

    A preset encapsulates a pre-configured set of validation requirements
    with a sampling strategy optimized for that use case. Presets reduce
    boilerplate by bundling common configurations.

    Attributes:
        requirements: List of Requirement instances to validate against.
        strategy: SamplingStrategy (typically RepairTemplateStrategy).
        feedback_strategy_name: Human-readable name for the feedback approach
            (e.g., "python_code_repair", "matplotlib_plotting_repair").
        description: Optional one-line description of the preset's purpose.
        example_usage: Optional usage example code snippet.
    """

    requirements: list[Requirement]
    strategy: SamplingStrategy
    feedback_strategy_name: str = "default"
    description: str | None = None
    example_usage: str | None = None


def python_code_generation_sampling(
    loop_budget: int = 2,
    *,
    allowed_imports: list[str] | None = None,
    output_limit_chars: int = 10_000,
    timeout_seconds: int = 5,
    use_sandbox: bool = False,
    feedback_quality: Literal["brief", "detailed"] = "detailed",
) -> SamplingPreset:
    """Pre-configured preset for Python code generation with repair feedback.

    Bundles python_code_generation_requirements() with a RepairTemplateStrategy
    optimized for Python code generation tasks. The strategy includes model-friendly
    feedback that converts validation failures into actionable repair instructions.

    Args:
        loop_budget: Maximum repair attempts. Default 2 (one retry + initial attempt).
        allowed_imports: Whitelist of allowed top-level module names. If None,
            all imports are allowed. Default None.
        output_limit_chars: Maximum captured stdout size in characters.
            Default 10,000. Prevents excessive logging or infinite loops.
        timeout_seconds: Maximum execution time per code sample in seconds.
            Default 5. Only enforced if use_sandbox=True.
        use_sandbox: Use Docker-isolated execution (llm-sandbox).
            Default False. Set to True for untrusted code generation.
        feedback_quality: Feedback verbosity level. Currently accepts "brief" and
            "detailed"; future versions may customize this. Default "detailed".

    Returns:
        SamplingPreset with bundled requirements and repair strategy.

    Raises:
        ValueError: If loop_budget < 1, or if timeout_seconds or output_limit_chars
            are not positive (raised by python_code_generation_requirements).

    Examples:
        Generate code with no import restrictions:

            >>> preset = python_code_generation_sampling()
            >>> result = session.instruct(
            ...     "Write a function to sum 1 to 100",
            ...     requirements=preset.requirements,
            ...     strategy=preset.strategy,
            ... )

        Generate code with strict import allowlist:

            >>> preset = python_code_generation_sampling(
            ...     allowed_imports=["numpy", "math"],
            ...     loop_budget=3,
            ... )
            >>> result = session.instruct(
            ...     "Write a function using numpy",
            ...     requirements=preset.requirements,
            ...     strategy=preset.strategy,
            ... )

        Generate code in isolated sandbox:

            >>> preset = python_code_generation_sampling(
            ...     use_sandbox=True,
            ...     timeout_seconds=10,
            ... )
    """
    requirements = python_code_generation_requirements(
        allowed_imports=allowed_imports,
        output_limit_chars=output_limit_chars,
        timeout_seconds=timeout_seconds,
        use_sandbox=use_sandbox,
    )

    strategy = ModelFriendlyRepairStrategy(
        loop_budget=loop_budget, requirements=requirements
    )

    return SamplingPreset(
        requirements=requirements,
        strategy=strategy,
        feedback_strategy_name="python_code_repair",
        description="Python code generation with smart repair feedback",
        example_usage=(
            "preset = python_code_generation_sampling(allowed_imports=['numpy'])\n"
            "result = session.instruct(prompt, requirements=preset.requirements, strategy=preset.strategy)"
        ),
    )


def python_plotting_sampling(
    output_path: str | None = None,
    loop_budget: int = 3,
    *,
    allowed_imports: list[str] | None = None,
    timeout_seconds: int = 10,
    use_sandbox: bool = True,
    feedback_quality: Literal["brief", "detailed"] = "detailed",
) -> SamplingPreset:
    """Pre-configured preset for matplotlib plotting with repair feedback.

    Extends python_code_generation_sampling() with plotting-specific constraints:
    - Requires matplotlib headless backend configuration (matplotlib.use('Agg'))
    - Validates plot file output if output_path is specified
    - More conservative defaults (higher loop_budget, sandbox enabled by default)

    This preset is optimized for generating matplotlib code from user prompts,
    with special attention to ensuring plots can render in headless environments.

    Args:
        output_path: Expected plot output file path to validate. If provided,
            a PlotFileSaved requirement is added. If None, only validates that
            plotting dependencies are available. Default None.
        loop_budget: Maximum repair attempts. Default 3 (more than code_generation,
            since plotting needs rendering iterations).
        allowed_imports: Whitelist of allowed module imports. Default None (all allowed).
        timeout_seconds: Maximum execution time per sample in seconds.
            Default 10 (higher than code generation due to rendering).
        use_sandbox: Use Docker-isolated execution. Default True (strongly recommended
            for untrusted plot generation to prevent malicious graphics operations).
        feedback_quality: Feedback verbosity. Default "detailed".

    Returns:
        SamplingPreset with bundled plotting requirements and repair strategy.

    Raises:
        ValueError: If loop_budget < 1, or if timeout_seconds or output_limit_chars
            are not positive.

    Examples:
        Generate a scatter plot:

            >>> preset = python_plotting_sampling(output_path="/tmp/scatter.png")
            >>> result = session.instruct(
            ...     "Create a scatter plot of random points",
            ...     requirements=preset.requirements,
            ...     strategy=preset.strategy,
            ... )

        Generate with strict import allowlist:

            >>> preset = python_plotting_sampling(
            ...     output_path="/tmp/plot.png",
            ...     allowed_imports=["matplotlib", "numpy"],
            ... )

        Minimal sandbox (for trusted code generation):

            >>> preset = python_plotting_sampling(
            ...     use_sandbox=False,
            ...     loop_budget=2,
            ... )
    """
    try:
        from ..requirements.plotting.matplotlib import (
            MatplotlibHeadlessBackend,
            PlotDependenciesAvailable,
            PlotFileSaved,
        )
    except ImportError as e:
        raise ImportError(
            "Matplotlib requirements not available. "
            "Install with: uv sync --extra plotting"
        ) from e

    python_reqs = python_code_generation_requirements(
        allowed_imports=allowed_imports,
        output_limit_chars=10_000,
        timeout_seconds=timeout_seconds,
        use_sandbox=use_sandbox,
    )

    plotting_reqs: list[Requirement] = [MatplotlibHeadlessBackend()]
    if output_path:
        plotting_reqs.append(PlotFileSaved(output_path=output_path))
    else:
        plotting_reqs.append(PlotDependenciesAvailable())

    all_reqs = python_reqs + plotting_reqs

    strategy = ModelFriendlyRepairStrategy(
        loop_budget=loop_budget, requirements=all_reqs
    )

    return SamplingPreset(
        requirements=all_reqs,
        strategy=strategy,
        feedback_strategy_name="matplotlib_plotting_repair",
        description="Matplotlib plotting with headless backend and repair feedback",
        example_usage=(
            "preset = python_plotting_sampling(output_path='/tmp/plot.png')\n"
            "result = session.instruct(prompt, requirements=preset.requirements, strategy=preset.strategy)"
        ),
    )
