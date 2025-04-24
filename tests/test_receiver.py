"""Tests for the Receiver class."""

import pytest
import astropy.units as u
from astropy.tests.helper import assert_quantity_allclose

from spacelink.receiver import Receiver
from spacelink.signal import Signal


def test_receiver_creation():
    """Test creating a Receiver with valid parameters."""
    receiver = Receiver(gain=20 * u.dB, noise_temperature=290 * u.K)
    assert_quantity_allclose(receiver.gain, 20 * u.dB)
    assert_quantity_allclose(receiver.noise_temperature, 290 * u.K)


def test_receiver_output():
    """Test that a Receiver produces the expected output signal."""
    receiver = Receiver(gain=20 * u.dB, noise_temperature=290 * u.K)
    
    # Set up input signal
    def input_signal():
        return Signal(power=1 * u.W, noise_temperature=290 * u.K)
    receiver.input = input_signal
    
    # Calculate output
    signal = receiver.output(1 * u.GHz)
    
    assert isinstance(signal, Signal)
    assert_quantity_allclose(signal.power, 100 * u.W)  # 20 dB gain = 100x power
    assert_quantity_allclose(signal.noise_temperature, 580 * u.K)  # 290K + 290K


def test_receiver_no_input():
    """Test that Receiver raises ValueError when input is not set."""
    receiver = Receiver(gain=20 * u.dB, noise_temperature=290 * u.K)
    with pytest.raises(ValueError, match="Receiver input must be set before calculating output"):
        receiver.output(1 * u.GHz)


def test_receiver_unit_conversion():
    """Test that Receiver handles unit conversions correctly."""
    # Create receiver with different gain unit
    receiver = Receiver(gain=20 * u.dB, noise_temperature=290 * u.K)
    assert_quantity_allclose(receiver.gain, 20 * u.dB)

    # Create receiver with different temperature unit
    receiver = Receiver(gain=20 * u.dB, noise_temperature=273.15 * u.K)  # 0°C in Kelvin
    assert_quantity_allclose(receiver.noise_temperature, 273.15 * u.K)


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