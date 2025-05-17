"""Tests for core antenna calculation functions."""

import pytest
import astropy.units as u
from astropy.tests.helper import assert_quantity_allclose

from spacelink.core.antenna import polarization_loss, dish_gain
from spacelink.core.units import Dimensionless, Length, Frequency, Decibels

"""
-VALIDATED-
This site was used to generate the following test cases:
https://phillipmfeldman.org/Engineering/pol_mismatch_loss.html
"""

def assert_decibel_equal(actual, expected, atol=1e-2):
    """
    Assert that two decibel quantities are equal within a tolerance, comparing value and unit string.
    Accepts dB, dB(1), Decibel, etc. Does not use .to() for conversion.
    """
    actual_val = actual.value if hasattr(actual, 'value') else actual
    expected_val = expected.value if hasattr(expected, 'value') else expected
    actual_unit = str(actual.unit) if hasattr(actual, 'unit') else str(actual)
    expected_unit = str(expected.unit) if hasattr(expected, 'unit') else str(expected)
    # Accept atol as float or Quantity
    if hasattr(atol, 'value'):
        atol = float(atol.value)
    assert actual_unit.startswith('dB') and expected_unit.startswith('dB'), f"Units must be dB-like, got {actual_unit} and {expected_unit}"
    assert abs(actual_val - expected_val) <= atol, f"Values differ: {actual_val} vs {expected_val} (atol={atol})"
    assert actual_unit == expected_unit or actual_unit.startswith('dB') and expected_unit.startswith('dB'), f"Unit strings differ: {actual_unit} vs {expected_unit}"

@pytest.mark.parametrize(
    "ar_tx_db, ar_rx_db, expected_loss_db, tol",
    [
        (0 * u.dB, 0 * u.dB, 0.0 * u.dB, 0.01 * u.dB),  # matchy matchy
        (0 * u.dB, 60 * u.dB, 3.002 * u.dB, 0.01 * u.dB),  # Circular to linear
        (60 * u.dB, 0 * u.dB, 3.002 * u.dB, 0.01 * u.dB),  # Linear to circular
        (1 * u.dB, 1 * u.dB, 0.057 * u.dB, 0.01 * u.dB),  # Small mismatch
        (3 * u.dB, 3 * u.dB, 0.508 * u.dB, 0.01 * u.dB),  # Small mismatch
        (1 * u.dB, 2 * u.dB, 0.128 * u.dB, 0.01 * u.dB),  # Small mismatch
        (2 * u.dB, 1 * u.dB, 0.128 * u.dB, 0.01 * u.dB),  # Small mismatch
        (2 * u.dB, 2 * u.dB, 0.228 * u.dB, 0.01 * u.dB),  # Small mismatch
        (2 * u.dB, 3 * u.dB, 0.354 * u.dB, 0.01 * u.dB),  # Small mismatch
        (3 * u.dB, 2 * u.dB, 0.354 * u.dB, 0.01 * u.dB),  # Small mismatch
        (3 * u.dB, 3 * u.dB, 0.508 * u.dB, 0.01 * u.dB),  # Small mismatch
        (3 * u.dB, 4 * u.dB, 0.684 * u.dB, 0.01 * u.dB),  # Small mismatch
        (4 * u.dB, 3 * u.dB, 0.684 * u.dB, 0.01 * u.dB),  # Small mismatch
        (4 * u.dB, 4 * u.dB, 0.890 * u.dB, 0.01 * u.dB),  # Small mismatch
        (4 * u.dB, 5 * u.dB, 1.114 * u.dB, 0.01 * u.dB),  # Small mismatch
        (5 * u.dB, 4 * u.dB, 1.114 * u.dB, 0.01 * u.dB),  # Small mismatch
        (5 * u.dB, 5 * u.dB, 1.366 * u.dB, 0.01 * u.dB),  # Small mismatch
    ],
)
def test_polarization_loss_calculation(ar_tx_db, ar_rx_db, expected_loss_db, tol):
    """
    -VALIDATED-
    Test the polarization loss calculation based on axial ratios.
    """
    assert_decibel_equal(
        polarization_loss(ar_tx_db, ar_rx_db), expected_loss_db, atol=tol
    )


# Unvalidated
@pytest.mark.parametrize(
    "diameter, frequency, efficiency, expected_gain_db, tol",
    [
        (20.0 * u.m, 8.4 * u.GHz, 0.65 * u.dimensionless, 63.04 * u.dB, 0.01 * u.dB),
        (1.0 * u.m, 2.4 * u.GHz, 0.65 * u.dimensionless, 26.14 * u.dB, 0.01 * u.dB),
        (1.0 * u.m, 2.4 * u.GHz, 0.5 * u.dimensionless, 25.00 * u.dB, 0.01 * u.dB),
        (20.0 * u.m, 2.4 * u.GHz, 0.65 * u.dimensionless, 52.16 * u.dB, 0.01 * u.dB),
        (1.0 * u.m, 5.8 * u.GHz, 0.65 * u.dimensionless, 33.80 * u.dB, 0.01 * u.dB),
        (1.0 * u.m, 900 * u.MHz, 0.65 * u.dimensionless, 17.62 * u.dB, 0.01 * u.dB),
    ],
)
def test_dish_gain_calculation(
    diameter: Length,
    frequency: Frequency,
    efficiency: Dimensionless,
    expected_gain_db: Decibels,
    tol: Decibels,
):
    """Test the core dish_gain calculation function with various parameters."""
    assert_decibel_equal(
        dish_gain(diameter, frequency, efficiency), expected_gain_db, atol=tol
    )


def test_dish_gain_invalid_frequency():
    """Test that dish_gain raises error for invalid frequency."""
    # Test invalid frequency
    with pytest.raises(ValueError, match="Frequency must be positive"):
        dish_gain(1.0 * u.m, 0 * u.Hz, 0.65 * u.dimensionless)
    with pytest.raises(ValueError, match="Frequency must be positive"):
        dish_gain(1.0 * u.m, -1 * u.GHz, 0.65 * u.dimensionless)
