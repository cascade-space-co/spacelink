"""Test configuration and fixtures."""

import pytest

import spacelink.core.units as units


@pytest.fixture(autouse=True)
def _enable_return_units_check():
    """Enable strict return unit checking during tests."""
    units._RETURN_UNITS_CHECK_ENABLED = True
    yield
    units._RETURN_UNITS_CHECK_ENABLED = False
