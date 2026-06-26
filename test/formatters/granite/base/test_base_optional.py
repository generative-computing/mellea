# SPDX-License-Identifier: Apache-2.0

import pytest

from mellea.formatters.granite.base.optional import nltk_check


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
