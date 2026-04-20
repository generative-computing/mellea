"""Shared fixtures for plugin tests."""

import pytest

from mellea.plugins.manager import has_plugins, shutdown_plugins


@pytest.fixture(autouse=True, scope="package")
async def _restore_plugins_after_package(request):
    """Re-register acceptance sets once after all plugin tests finish.

    Plugin tests freely shut down the manager for isolation.  This package-scoped
    fixture captures whether plugins were active at the start and restores them in
    teardown so that test modules collected *after* ``test/plugins/`` still see the
    session-scoped acceptance sets registered by ``auto_register_acceptance_sets``.
    """
    plugins_disabled = request.config.getoption(
        "--disable-default-mellea-plugins", default=False
    )
    was_enabled = has_plugins()
    yield
    if was_enabled and not plugins_disabled:
        from mellea.plugins import register
        from test.plugins._acceptance_sets import ALL_ACCEPTANCE_SETS

        for ps in ALL_ACCEPTANCE_SETS:
            register(ps)


@pytest.fixture(autouse=True)
async def reset_plugins():
    """Shut down plugins before and after each test for isolation."""
    await shutdown_plugins()
    yield
    await shutdown_plugins()
