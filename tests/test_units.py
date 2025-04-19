"""Tests for the units module."""

import pytest
from pint.testing import assert_allclose
from pyradio.units import (
    Q_,
    Hz,
    kHz,
    MHz,
    GHz,
    W,
    mW,
    m,
    km,
    SPEED_OF_LIGHT,
    return_loss_to_vswr,
    vswr_to_return_loss,
    wavelength,
    frequency,
    db,
)


def test_unit_conversion():
    """Test basic unit conversions."""
    # Frequency conversions
    assert_allclose((1 * GHz).to(MHz), 1000 * MHz)
    assert_allclose((1 * MHz).to(kHz), 1000 * kHz)
    assert_allclose((1 * kHz).to(Hz), 1000 * Hz)

    # Power conversions
    assert_allclose((1 * W).to(mW), 1000 * mW)

    assert_allclose(Q_(0.0, "dBW").to("W"), 1 * W)
    assert_allclose(Q_(30, "dBm").to("W"), 1 * W)

    # Distance conversions
    assert_allclose((1 * km).to(m), 1000 * m)


def test_speed_of_light():
    """Test the speed of light constant."""
    assert_allclose(SPEED_OF_LIGHT, Q_(299792458.0, "meter/second"))
    assert str(SPEED_OF_LIGHT.units) == "meter / second"


def test_wavelength_calculation():
    """Test wavelength calculation from frequency."""
    # Test at different frequencies
    assert_allclose(wavelength(1 * GHz).to(m), 0.299792458 * m)
    assert_allclose(wavelength(300 * MHz).to(m), 0.999308193 * m)
    assert_allclose(wavelength(30 * kHz).to(m), 9993.08193 * m)

    # Test unit conversion in result
    wavelength_result = wavelength(2.4 * GHz)
    assert_allclose(wavelength_result.to(m), 0.12491352416667 * m)


def test_frequency_calculation():
    """Test frequency calculation from wavelength."""
    # Test at different wavelengths
    assert_allclose(frequency(1 * m).to(MHz), 299.792458 * MHz)
    assert_allclose(frequency(10 * m).to(MHz), 29.9792458 * MHz)
    assert_allclose(frequency(0.1 * m).to(GHz), 2.99792458 * GHz)

    # Test unit conversion in result
    freq_result = frequency(0.125 * m)
    assert_allclose(freq_result.to(GHz), 2.39833966 * GHz)


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
        wavelength(1.0 * m)
    # Frequency with non-length input
    with pytest.raises(Exception):
        frequency(1 * Hz)


def test_vswr():
    test_data = [
        (1.1, 0.0476, 26.44),
        (1.2, 0.0909, 20.83),
        (1.3, 0.1304, 17.69),
        (1.4, 0.1667, 15.56),
        (1.5, 0.2000, 13.98),
        (1.6, 0.2308, 12.74),
        (1.7, 0.2593, 11.73),
        (1.8, 0.2857, 10.88),
        (1.9, 0.3103, 10.16),
        (2.0, 0.3333, 9.54),
    ]
    for vswr, gamma, return_loss in test_data:
        assert return_loss_to_vswr(return_loss) == pytest.approx(vswr, abs=0.01)
        assert vswr_to_return_loss(vswr) == pytest.approx(return_loss, abs=0.01)
