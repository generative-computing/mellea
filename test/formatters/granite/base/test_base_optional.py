# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from mellea.formatters.granite.base import optional, util
from mellea.formatters.granite.base.optional import import_optional, nltk_check


class TestImportOptional:
    """Verify import_optional re-raises ImportError and is single-sourced."""

    def test_reraises_import_error(self):
        with pytest.raises(ImportError):
            with import_optional("hf"):
                raise ImportError("No module named 'torch'")

    def test_no_error_passes_through(self):
        with import_optional("hf"):
            pass  # no exception — should succeed silently

    def test_util_reexports_optional(self):
        # Single source of truth: util must re-export optional's object,
        # not define its own duplicate.
        assert util.import_optional is optional.import_optional


class TestNltkCheck:
    """Verify nltk_check catches both ImportError and LookupError."""

    def test_import_error_gives_install_hint(self):
        with pytest.raises(ImportError, match="mellea"):
            with nltk_check("citation parsing"):
                raise ImportError("No module named 'nltk'")

    def test_lookup_error_gives_install_hint(self):
        with pytest.raises(ImportError, match="punkt_tab"):
            with nltk_check("citation parsing"):
                raise LookupError("Resource punkt_tab not found")

    def test_no_error_passes_through(self):
        with nltk_check("citation parsing"):
            pass  # no exception — should succeed silently
