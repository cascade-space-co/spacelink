"""Tests for the Receiver module."""

import pytest
import astropy.units as u
from astropy.tests.helper import assert_quantity_allclose

from spacelink.receiver import Receiver
from spacelink.signal import Signal
from spacelink.source import Source


class MockSource(Source):
    """Concrete implementation of Source for testing."""
    def __init__(self):
        super().__init__()
        self._power = 1 * u.W
        self.noise_temperature = 290 * u.K

    def output(self, frequency):
        return Signal(power=self._power, noise_temperature=self.noise_temperature)


def test_receiver_creation():
    """Test creating a Receiver with valid parameters."""
    receiver = Receiver(gain=20 * u.dB, noise_temperature=290 * u.K)
    assert_quantity_allclose(receiver.gain(1 * u.GHz), 20 * u.dB)
    assert_quantity_allclose(receiver.noise_temperature, 290 * u.K)


def test_receiver_output():
    """Test that a Receiver produces the expected output signal."""
    receiver = Receiver(gain=20 * u.dB, noise_temperature=290 * u.K)
    
    # Set up input signal
    source = MockSource()
    receiver.input = source
    
    # Get output
    output = receiver.output(1 * u.GHz)
    
    # Check that power is amplified by gain
    assert_quantity_allclose(output.power, 100 * u.W)  # 20 dB = 100x power
    
    # Check that noise temperature is sum of input and receiver
    assert_quantity_allclose(output.noise_temperature, 580 * u.K)


def test_receiver_no_input():
    """Test that Receiver raises ValueError when input is not set."""
    receiver = Receiver(gain=20 * u.dB, noise_temperature=290 * u.K)
    
    with pytest.raises(ValueError, match="Receiver input must be set before calculating output"):
        receiver.output(1 * u.GHz)


def test_receiver_unit_conversion():
    """Test that Receiver handles unit conversions correctly."""
    # Create receiver with different gain unit
    receiver = Receiver(gain=20 * u.dB, noise_temperature=290 * u.K)
    assert_quantity_allclose(receiver.gain(1 * u.GHz), 20 * u.dB)
    
    # Create receiver with different temperature unit
    receiver = Receiver(gain=20 * u.dB, noise_temperature=17 * u.deg_C)
    assert_quantity_allclose(receiver.noise_temperature, 290.15 * u.K)


def test_receiver_invalid_gain_type():
    """Test that creating a Receiver with non-Quantity gain raises TypeError."""
    with pytest.raises(TypeError, match="Gain must be a Quantity with dB units"):
        Receiver(gain=20.0, noise_temperature=290 * u.K)


def test_receiver_invalid_temperature_type():
    """Test that creating a Receiver with non-Quantity temperature raises TypeError."""
    with pytest.raises(TypeError, match="Noise temperature must be a Quantity with temperature units"):
        Receiver(gain=20 * u.dB, noise_temperature=290.0)


def test_receiver_invalid_gain_unit():
    """Test that creating a Receiver with wrong gain unit raises UnitConversionError."""
    with pytest.raises(u.UnitConversionError):
        Receiver(gain=20 * u.W, noise_temperature=290 * u.K)


def test_receiver_invalid_temperature_unit():
    """Test that creating a Receiver with wrong temperature unit raises UnitConversionError."""
    with pytest.raises(u.UnitConversionError):
        Receiver(gain=20 * u.dB, noise_temperature=290 * u.m)


def test_receiver_negative_temperature():
    """Test that creating a Receiver with negative temperature raises ValueError."""
    with pytest.raises(ValueError, match="Noise temperature must be non-negative"):
        Receiver(gain=20 * u.dB, noise_temperature=-290 * u.K) 