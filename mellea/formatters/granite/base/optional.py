# SPDX-License-Identifier: Apache-2.0

"""Context-manager helpers for gracefully handling optional import dependencies.

Provides ``import_optional``, a context manager that catches ``ImportError`` and
re-raises it with a human-readable install hint (e.g. ``pip install <package>[extra]``),
and ``nltk_check``, a variant tailored to NLTK data-download errors. Used by Granite
formatter modules that have optional third-party dependencies.
"""

# Standard
import logging
from contextlib import contextmanager

_NLTK_INSTALL_INSTRUCTIONS = """
Please install nltk with:
    pip install nltk
In some environments you may also need to manually download model weights with:
    python -m nltk.downloader punkt_tab
See https://www.nltk.org/install.html#installing-nltk-data for more detailed
instructions."""


@contextmanager
def import_optional(extra_name: str):
    """Handle optional imports.

    Args:
        extra_name: Package extra to suggest in the install hint
            (e.g. ``pip install mellea[extra_name]``).
    """
    try:
        yield
    except ImportError as err:
        logging.warning(
            "%s.\nHINT: You may need to pip install %s[%s]",
            err,
            __package__,
            extra_name,
        )
        raise


@contextmanager
def nltk_check(feature_name: str):
    """Variation on import_optional for nltk.

    Args:
        feature_name: Name of the feature that requires NLTK, used in the error message.

    Raises:
        ImportError: If the ``nltk`` package is not installed or required
            NLTK data (e.g. ``punkt_tab``) has not been downloaded,
            re-raised with a descriptive message and installation
            instructions.
    """
    try:
        yield
    except ImportError as err:
        raise ImportError(
            f"'nltk' package not installed. This package is required for "
            f"{feature_name} in the 'mellea' library."
            f"{_NLTK_INSTALL_INSTRUCTIONS}"
        ) from err
    except LookupError as err:
        raise ImportError(
            f"NLTK data required for {feature_name} is not installed."
            f"{_NLTK_INSTALL_INSTRUCTIONS}"
        ) from err
