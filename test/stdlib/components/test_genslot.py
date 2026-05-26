"""Tests for the backward-compatibility genslot shim module."""

import importlib
import sys
import warnings

# -- Module-level deprecation warning --


def test_import_emits_deprecation_warning():
    sys.modules.pop("mellea.stdlib.components.genslot", None)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.import_module("mellea.stdlib.components.genslot")

    dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(dep_warnings) == 1
    assert "genslot has been renamed" in str(dep_warnings[0].message)
    assert "genstub" in str(dep_warnings[0].message)


def test_warning_message_mentions_update():
    sys.modules.pop("mellea.stdlib.components.genslot", None)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.import_module("mellea.stdlib.components.genslot")

    msg = str(caught[0].message)
    assert "update your imports" in msg.lower()


# -- Old class names accessible --


def test_generative_slot_accessible():
    from mellea.stdlib.components.genslot import GenerativeSlot

    assert GenerativeSlot is not None


def test_sync_generative_slot_accessible():
    from mellea.stdlib.components.genslot import SyncGenerativeSlot

    assert SyncGenerativeSlot is not None


def test_async_generative_slot_accessible():
    from mellea.stdlib.components.genslot import AsyncGenerativeSlot

    assert AsyncGenerativeSlot is not None


# -- Alias identity --


def test_generative_slot_is_generative_stub():
    from mellea.stdlib.components.genslot import GenerativeSlot
    from mellea.stdlib.components.genstub import GenerativeStub

    assert GenerativeSlot is GenerativeStub


def test_sync_generative_slot_is_sync_generative_stub():
    from mellea.stdlib.components.genslot import SyncGenerativeSlot
    from mellea.stdlib.components.genstub import SyncGenerativeStub

    assert SyncGenerativeSlot is SyncGenerativeStub


def test_async_generative_slot_is_async_generative_stub():
    from mellea.stdlib.components.genslot import AsyncGenerativeSlot
    from mellea.stdlib.components.genstub import AsyncGenerativeStub

    assert AsyncGenerativeSlot is AsyncGenerativeStub


# -- Core re-exports --


def test_precondition_exception_reexported():
    from mellea.stdlib.components.genslot import PreconditionException
    from mellea.stdlib.components.genstub import PreconditionException as PE

    assert PreconditionException is PE


def test_generative_decorator_reexported():
    from mellea.stdlib.components.genslot import generative
    from mellea.stdlib.components.genstub import generative as gen

    assert generative is gen
