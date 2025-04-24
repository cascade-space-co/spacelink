"""Tests for the Transmitter class."""

import pytest
import astropy.units as u
from astropy.tests.helper import assert_quantity_allclose

from spacelink.transmitter import Transmitter
from spacelink.signal import Signal


def test_transmitter_creation():
    """Test creating a Transmitter with valid parameters."""
    transmitter = Transmitter(power=1 * u.W, noise_temperature=290 * u.K)
    assert_quantity_allclose(transmitter.power, 1 * u.W)
    assert_quantity_allclose(transmitter.noise_temperature, 290 * u.K)


def test_transmitter_output():
    """Test that a Transmitter produces the expected output signal."""
    transmitter = Transmitter(power=1 * u.W, noise_temperature=290 * u.K)
    signal = transmitter.output(1 * u.GHz)
    
    assert isinstance(signal, Signal)
    assert_quantity_allclose(signal.power, 1 * u.W)
    assert_quantity_allclose(signal.noise_temperature, 290 * u.K)
