"""Plotting-specific requirements for Python tool validation.

Provides matplotlib and plotting-focused requirement factories separate from
generic Python tool requirements.
"""

from .matplotlib import python_plotting_requirements

__all__ = ["python_plotting_requirements"]
