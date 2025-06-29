"""Tests for the units module."""

import astropy.units as u
from astropy.tests.helper import assert_quantity_allclose
import pytest
import numpy as np

from spacelink.core import units
from spacelink.core.units import (
    return_loss_to_vswr,
    vswr_to_return_loss,
    wavelength,
    frequency,
    to_dB,
)


@pytest.mark.parametrize(
    "frequency, expected_wavelength",
    [
        (1 * u.GHz, 0.299792458 * u.m),
        (300 * u.MHz, 0.999308193 * u.m),
        (30 * u.kHz, 9993.08193 * u.m),
    ],
)
def test_wavelength_calculation(frequency, expected_wavelength):
    """
    TODO: validate
    """
    assert_quantity_allclose(wavelength(frequency).to(u.m), expected_wavelength)


# DO NOT MODIFY
@pytest.mark.parametrize(
    "wavelength, expected_frequency",
    [
        (1 * u.m, 299.792458 * u.MHz),
        (10 * u.m, 29.9792458 * u.MHz),
        (0.1 * u.m, 2.99792458 * u.GHz),
    ],
)
def test_frequency_calculation(wavelength, expected_frequency):
    """
    TODO: validate
    """
    assert_quantity_allclose(
        frequency(wavelength).to(expected_frequency.unit), expected_frequency
    )


@pytest.mark.parametrize(
    "func,invalid_input,error_type",
    [
        (wavelength, 1 * u.m, u.UnitConversionError),
        (wavelength, 1.0 * u.m, u.UnitConversionError),
        (frequency, 1 * u.Hz, u.UnitConversionError),
    ],
)
def test_invalid_inputs(func, invalid_input, error_type):
    with pytest.raises(error_type):
        func(invalid_input)


def assert_decibel_equal(actual, expected, atol=0.01):
    # Compare value
    assert np.isclose(
        actual.to_value(expected.unit), expected.value, atol=atol
    ), f"{actual} != {expected}"
    # Compare unit string (should both be 'dB')
    assert str(actual.unit) == str(
        expected.unit
    ), f"Units differ: {actual.unit} != {expected.unit}"


# DO NOT MODIFY
@pytest.mark.parametrize(
    "vswr,gamma,return_loss",
    [
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
    ],
)
def test_vswr(vswr, gamma, return_loss):
    """
    -VALIDATED-
    """
    # Test return loss to VSWR conversion
    vswr_result = return_loss_to_vswr(return_loss * u.dB)
    assert_quantity_allclose(
        vswr_result, vswr * u.dimensionless, atol=0.01 * u.dimensionless
    )

    # Test VSWR to return loss conversion (use safe_negate)
    gamma_q = gamma * u.dimensionless
    return_loss_result = safe_negate(to_dB(gamma_q, factor=20))
    assert_decibel_equal(return_loss_result, return_loss * u.dB, atol=0.01)


def test_enforce_units_decorator():
    """Test that enforce_units decorator raises the correct astropy exception
    with incompatible units."""
    with pytest.raises(u.UnitConversionError):
        wavelength(1.0 * u.m)

    with pytest.raises(u.UnitConversionError):
        frequency(1.0 * u.Hz)


# DO NOT MODIFY
@pytest.mark.parametrize(
    "input_value, factor, expected",
    [
        (100 * u.dimensionless, 10, 20 * u.dB),  # dB for ratios
        (1000 * u.dimensionless, 10, 30 * u.dB),
        (10 * u.dimensionless, 20, 20 * u.dB),
    ],
)
def test_to_dB(input_value, factor, expected):
    """
    -VALIDATED-
    """
    assert_decibel_equal(units.to_dB(input_value, factor=factor), expected, atol=0.01)


@pytest.mark.parametrize(
    "input_value, factor, expected",
    [
        (20 * u.dB, 10, 100 * u.dimensionless),
        (30 * u.dB, 10, 1000 * u.dimensionless),
        (20 * u.dB, 20, 10 * u.dimensionless),
    ],
)
def test_to_linear(input_value, factor, expected):
    """Test conversion from decibels to linear."""
    assert_quantity_allclose(units.to_linear(input_value, factor=factor), expected)


@pytest.mark.parametrize(
    "return_loss, vswr",
    [
        (20 * u.dB, 1.222 * u.dimensionless),
        (float("inf") * u.dB, 1.0 * u.dimensionless),
    ],
)
def test_vswr_return_loss_conversions(return_loss, vswr):
    """
    TODO: validate
    """
    vswr_result = units.return_loss_to_vswr(return_loss)
    assert_quantity_allclose(vswr_result, vswr, atol=0.01 * u.dimensionless)

    # Use safe_negate for dB quantities
    gamma = (vswr - 1) / (vswr + 1)
    return_loss_result = safe_negate(to_dB(gamma, factor=20))
    assert_decibel_equal(return_loss_result, return_loss, atol=0.01)


def test_return_loss_to_vswr_invalid_input():
    with pytest.raises(ValueError):
        units.return_loss_to_vswr(-1 * u.dB)


def test_vswr_to_return_loss_invalid_input():
    with pytest.raises(ValueError):
        vswr_to_return_loss(0.5 * u.dimensionless)


@pytest.mark.parametrize(
    "input_value, factor, expected",
    [
        (1 * u.W, 10, 0 * u.dBW),
        (10 * u.W, 10, 10 * u.dBW),
        (1 * u.K, 10, 0 * u.dBK),
        (100 * u.K, 10, 20 * u.dBK),
        (1 * u.Hz, 10, 0 * u.dBHz),
        (1000 * u.Hz, 10, 30 * u.dBHz),
        (100 * u.dimensionless, 10, 20 * u.dB),
        (10 * u.dimensionless, 20, 20 * u.dB),
    ],
)
def test_to_dB_units(input_value, factor, expected):
    """Test to_dB preserves logarithmic units (dBW, dBK, dBHz, dB)."""
    if input_value.unit == u.dimensionless:
        assert_decibel_equal(
            units.to_dB(input_value, factor=factor), expected, atol=0.01
        )
    else:
        assert_quantity_allclose(
            units.to_dB(input_value, factor=factor), expected, atol=0.01 * expected.unit
        )


def safe_negate(quantity):
    # Astropy does not allow -quantity for function units, so use multiplication
    return (-1) * quantity


def test_to_dB_output_unit_is_dBW():
    """Test that to_dB with input in W returns output in dBW."""
    value = 5.0
    result = units.to_dB(value * u.W)
    assert result.unit == u.dBW, f"Expected unit dBW, got {result.unit}"
