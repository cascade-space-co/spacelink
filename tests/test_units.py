"""Tests for the units module."""

import pytest
from pyradio.units import (
    Q_, Hz, kHz, MHz, GHz,
    W, mW,
    m, km, K,
    dimensionless,
    SPEED_OF_LIGHT,
    wavelength, frequency, db
)


def test_unit_conversion():
    """Test basic unit conversions."""
    # Frequency conversions
    assert (1 * GHz).to(MHz).magnitude == pytest.approx(1000)
    assert (1 * MHz).to(kHz).magnitude == pytest.approx(1000)
    assert (1 * kHz).to(Hz).magnitude == pytest.approx(1000)

    # Power conversions
    assert (1 * W).to(mW).magnitude == pytest.approx(1000)

    assert Q_(0.0, 'dBW').to('W').magnitude == pytest.approx(1.0)
    assert Q_(30, 'dBm').to('W').magnitude == pytest.approx(1.0)

    # Distance conversions
    assert (1 * km).to(m).magnitude == pytest.approx(1000)


def test_speed_of_light():
    """Test the speed of light constant."""
    assert SPEED_OF_LIGHT.magnitude == pytest.approx(299792458.0)
    assert str(SPEED_OF_LIGHT.units) == 'meter / second'


def test_wavelength_calculation():
    """Test wavelength calculation from frequency."""
    # Test at different frequencies
    assert wavelength(1 * GHz).to(m).magnitude == pytest.approx(0.299792458)
    assert wavelength(300 * MHz).to(m).magnitude == pytest.approx(0.999308193)
    assert wavelength(30 * kHz).to(m).magnitude == pytest.approx(9993.08193)

    # Test unit conversion in result
    wavelength_result = wavelength(2.4 * GHz)
    assert wavelength_result.to(m).magnitude == pytest.approx(0.12491352416667)


def test_frequency_calculation():
    """Test frequency calculation from wavelength."""
    # Test at different wavelengths
    assert frequency(1 * m).to(MHz).magnitude == pytest.approx(299.792458)
    assert frequency(10 * m).to(MHz).magnitude == pytest.approx(29.9792458)
    assert frequency(0.1 * m).to(GHz).magnitude == pytest.approx(2.99792458)

    # Test unit conversion in result
    freq_result = frequency(0.125 * m)
    assert freq_result.to(GHz).magnitude == pytest.approx(2.39833966)


def test_db_conversion():
    """Test dB conversion function."""
    assert db(1.0) == pytest.approx(0.0)
    assert db(10.0) == pytest.approx(10.0)
    assert db(100.0) == pytest.approx(20.0)
    assert db(0.1) == pytest.approx(-10.0)
    assert db(0.01) == pytest.approx(-20.0)


def test_invalid_inputs():
    """Test the functions raise errors with invalid inputs."""
    # Wavelength with non-frequency input
    with pytest.raises(Exception):
        wavelength(1 * m)

    with pytest.raises(Exception):
        wavelength(1.0*m)
    # Frequency with non-length input
    with pytest.raises(Exception):
        frequency(1 * Hz)
