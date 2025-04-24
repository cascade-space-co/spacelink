"""Tests for the Signal class."""

import pytest
import astropy.units as u
from astropy.tests.helper import assert_quantity_allclose

from spacelink.signal import Signal


def test_signal_creation():
    """Test creating a Signal with valid parameters."""
    signal = Signal(power=1 * u.W, noise_temperature=290 * u.K)
    assert_quantity_allclose(signal.power, 1 * u.W)
    assert_quantity_allclose(signal.noise_temperature, 290 * u.K)




