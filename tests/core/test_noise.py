"""Tests for the noise module."""

import pytest
from typing import Tuple, List
import astropy.units as u
from astropy.tests.helper import assert_quantity_allclose
from spacelink.core import noise
from spacelink.core.units import Dimensionless, Decibels, Temperature

# Define type aliases for clarity
LinearStage = Tuple[Dimensionless, Dimensionless]
DbStage = Tuple[Decibels, Decibels]
# from spacelink.core import units # Unused


def test_thermal_noise_power():
    """Test thermal noise power calculation."""
    # Test with standard room temperature (290K)
    expected = 1.380649e-23 * 290.0 * 1e6
    assert_quantity_allclose(noise.noise_power(1.0 * u.MHz), expected * u.W)

    # Test with different temperature
    expected = 1.380649e-23 * 100.0 * 1e6
    assert_quantity_allclose(noise.noise_power(1.0 * u.MHz, 100 * u.K), expected * u.W)

    # Test with zero bandwidth
    assert_quantity_allclose(noise.noise_power(0.0 * u.MHz), 0.0 * u.W)

    # Test with negative bandwidth
    with pytest.raises(ValueError):
        noise.noise_power(-1.0 * u.MHz)


# DO NOT MODIFY - This test uses validated reference values
@pytest.mark.parametrize(
    "temperature, bandwidth, expected_noise_dBW",
    [
        (50 * u.K, 10 * u.Hz, -201.61 * u.dBW),
        (100 * u.K, 10 * u.Hz, -198.60 * u.dBW),
        (150 * u.K, 10 * u.Hz, -196.84 * u.dBW),
        (200 * u.K, 10 * u.Hz, -195.59 * u.dBW),
        (290 * u.K, 10 * u.Hz, -193.98 * u.dBW),
    ],
)
def test_noise_dBW_conversion(temperature, bandwidth, expected_noise_dBW):
    """
    -VALIDATED-
    """
    noise_w = noise.noise_power(bandwidth, temperature)
    assert_quantity_allclose(
        noise_w.to(u.dB(u.W)), expected_noise_dBW, atol=0.01 * u.dB(u.W)
    )


# Parameterized test with only the validated case
@pytest.mark.parametrize(
    "temperature, expected_noise_figure",
    [
        (290 * u.K, 3.0103 * u.dB),  # Standard room temperature - Validated
        # Add more validated cases here if available
    ],
)
def test_temperature_to_noise_figure(temperature, expected_noise_figure):
    """Test temperature to noise figure conversion."""  # Updated docstring
    nf = noise.temperature_to_noise_figure(temperature)
    assert_quantity_allclose(nf, expected_noise_figure, atol=0.01 * u.dB)


# TODO: test cascaded noise values
TempStage = Tuple[Temperature, Dimensionless]


@pytest.mark.parametrize(
    "stages, expected_nf",
    [
        # UNVALIDATED TEST DATA - Single stage
        ([(2.0 * u.dimensionless, 10.0 * u.dimensionless)], 2.0 * u.dimensionless),
        # UNVALIDATED TEST DATA - Two stages
        (
            [
                (2.0 * u.dimensionless, 10.0 * u.dimensionless),
                (3.0 * u.dimensionless, 5.0 * u.dimensionless),
            ],
            2.2 * u.dimensionless,
        ),
        # UNVALIDATED TEST DATA - Three stages
        (
            [
                (2.0 * u.dimensionless, 10.0 * u.dimensionless),
                (3.0 * u.dimensionless, 5.0 * u.dimensionless),
                (4.0 * u.dimensionless, 2.0 * u.dimensionless),
            ],
            2.26 * u.dimensionless,
        ),
        # Add more test cases as needed, ensuring data is validated
    ],
)
def test_cascaded_noise_factor(stages: List[LinearStage], expected_nf: Dimensionless):
    """Test cascaded_noise_factor calculation."""
    total_nf = noise.cascaded_noise_factor(stages)
    assert_quantity_allclose(total_nf, expected_nf, rtol=1e-6)


# Test cascaded_noise_factor with empty list
def test_cascaded_noise_factor_empty():
    """Test cascaded_noise_factor with empty stages list."""
    with pytest.raises(ValueError):
        noise.cascaded_noise_factor([])


@pytest.mark.parametrize(
    "stages, expected_nf_db",
    [
        # UNVALIDATED TEST DATA - Single stage
        # NF1=3dB (F=2), G1=10dB (G=10)
        ([(3.0 * u.dB, 10.0 * u.dB)], 3.0 * u.dB),
        # UNVALIDATED TEST DATA - Two stages
        # NF1=3dB (F=2), G1=10dB (G=10)
        # NF2=4.77dB (F=3), G2=7dB (G=5)
        # F_tot = 2 + (3-1)/10 = 2.2 -> NF_tot = 10*log10(2.2) = 3.424 dB
        ([(3.0 * u.dB, 10.0 * u.dB), (4.77 * u.dB, 7.0 * u.dB)], 3.424 * u.dB),
        # UNVALIDATED TEST DATA - Three stages
        # NF1=3dB (F=2), G1=10dB (G=10)
        # NF2=4.77dB (F=3), G2=7dB (G=5)
        # NF3=6.02dB (F=4), G3=3dB (G=2)
        # F_tot = 2.2 + (4-1)/(10*5) = 2.2 + 3/50 = 2.26 -> NF_tot = 10*log10(2.26) = 3.541 dB
        (
            [
                (3.0 * u.dB, 10.0 * u.dB),
                (4.77 * u.dB, 7.0 * u.dB),
                (6.02 * u.dB, 3.0 * u.dB),
            ],
            3.541 * u.dB,
        ),
        # Add more test cases as needed, ensuring data is validated
    ],
)
def test_cascaded_noise_figure(stages: List[DbStage], expected_nf_db: Decibels):
    """Test cascaded_noise_figure calculation."""
    total_nf_db = noise.cascaded_noise_figure(stages)
    assert_quantity_allclose(total_nf_db, expected_nf_db, atol=0.01 * u.dB)


# Test cascaded_noise_figure with empty list
def test_cascaded_noise_figure_empty():
    """Test cascaded_noise_figure with empty stages list."""
    with pytest.raises(ValueError):
        noise.cascaded_noise_figure([])


@pytest.mark.parametrize(
    "stages, expected_temp",
    [
        # UNVALIDATED TEST DATA - Single stage
        # T1=290K (F=2), G1=10 (G=10dB)
        ([(290 * u.K, 10.0 * u.dimensionless)], 290 * u.K),
        # UNVALIDATED TEST DATA - Two stages
        # T1=290K (F=2), G1=10 (G=10dB)
        # T2=580K (F=3), G2=5 (G=7dB)
        # F_tot = 2 + (3-1)/10 = 2.2 -> T_tot = (2.2 - 1) * 290 = 1.2 * 290 = 348 K
        (
            [
                (290 * u.K, 10.0 * u.dimensionless),
                (580 * u.K, 5.0 * u.dimensionless),
            ],
            348 * u.K,
        ),
        # UNVALIDATED TEST DATA - Three stages
        # T1=290K (F=2), G1=10 (G=10dB)
        # T2=580K (F=3), G2=5 (G=7dB)
        # T3=870K (F=4), G3=2 (G=3dB)
        # F_tot = 2.2 + (4-1)/(10*5) = 2.26 -> T_tot = (2.26 - 1) * 290 = 1.26 * 290 = 365.4 K
        (
            [
                (290 * u.K, 10.0 * u.dimensionless),
                (580 * u.K, 5.0 * u.dimensionless),
                (870 * u.K, 2.0 * u.dimensionless),
            ],
            365.4 * u.K,
        ),
        # Add more test cases as needed, ensuring data is validated
    ],
)
def test_cascaded_noise_temperature(
    stages: List[TempStage], expected_temp: Temperature
):
    """Test cascaded_noise_temperature calculation."""
    total_temp = noise.cascaded_noise_temperature(stages)
    assert_quantity_allclose(total_temp, expected_temp, rtol=1e-6)


# Test cascaded_noise_temperature with empty list
def test_cascaded_noise_temperature_empty():
    """Test cascaded_noise_temperature with empty stages list."""
    with pytest.raises(ValueError):
        noise.cascaded_noise_temperature([])
