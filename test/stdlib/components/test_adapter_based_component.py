"""Unit tests for the AdapterBasedComponent placeholder (issue #1134)."""

from mellea.stdlib.components.adapter_based_component import AdapterBasedComponent
from mellea.stdlib.components.intrinsic import Intrinsic


def test_adapter_based_component_is_intrinsic():
    assert AdapterBasedComponent is Intrinsic


def test_old_import_path_still_works():
    from mellea.stdlib.components.intrinsic import Intrinsic as _Intrinsic

    assert _Intrinsic is AdapterBasedComponent
