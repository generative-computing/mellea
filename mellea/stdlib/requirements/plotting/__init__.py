"""Matplotlib-specific requirements for validating plotting code.

This module provides requirements for validating Python code that generates
plots using matplotlib. Requirements validate that:
- matplotlib is configured with a headless backend
- plots are explicitly saved to files
- required dependencies (matplotlib, numpy) are importable
"""

from .matplotlib import (
    MatplotlibHeadlessBackend,
    PlotDependenciesAvailable,
    PlotFileSaved,
    python_plotting_requirements,
)

__all__ = [
    "MatplotlibHeadlessBackend",
    "PlotDependenciesAvailable",
    "PlotFileSaved",
    "python_plotting_requirements",
]
