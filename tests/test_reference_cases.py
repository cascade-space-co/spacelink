"""
Tests that verify the consistency of the reference test cases.

These tests demonstrate how to use the reference test cases for
verifying various functions across the codebase.
"""

import pytest
import astropy.units as u
from spacelink.units import wavelength
from test_cases import load_test_case, list_test_cases


def test_wavelength_reference():
    """Test wavelength calculation using reference test cases."""
    # Load the deep space case
    case = load_test_case("deep_space")

    # Calculate wavelength using the wavelength function
    calc_wavelength = wavelength(case.frequency)

    # Verify against reference value using pytest.approx for comparing quantity magnitudes
    assert calc_wavelength.to("m").value == pytest.approx(
        case.ref.wavelength.to("m").value, rel=0.001
    )

    # Load the lunar downlink case
    case = load_test_case("lunar_downlink")

    # Calculate wavelength using the wavelength function
    calc_wavelength = wavelength(case.frequency)

    # Verify against reference value using pytest.approx for comparing quantity magnitudes
    assert calc_wavelength.to("m").value == pytest.approx(
        case.ref.wavelength.to("m").value, rel=0.001
    )


def test_dish_parameters():
    """Test dish parameters from test cases."""
    # Load deep space case with dish antennas
    case = load_test_case("deep_space")

    # Verify dish parameters are set correctly
    assert case.tx_dish_diameter > 0 * u.m
    assert case.tx_dish_efficiency > 0 * u.linear
    assert case.tx_dish_efficiency <= 1 * u.linear

    # For dB values
    assert case.tx_antenna_gain > 0 * u.dB
    assert case.tx_antenna_axial_ratio >= 0 * u.dB

    # Same for receiver
    assert case.rx_dish_diameter > 0 * u.m
    assert case.rx_dish_efficiency > 0 * u.linear
    assert case.rx_dish_efficiency <= 1 * u.linear
    assert case.rx_antenna_gain > 0 * u.dB
    assert case.rx_antenna_axial_ratio >= 0 * u.dB


def test_wavelength_calculation():
    """Test wavelength computation with test cases."""
    # Test with multiple cases
    cases = ["deep_space", "lunar_uplink", "leo_downlink"]

    for case_name in cases:
        case = load_test_case(case_name)
        # Compare with units directly
        assert case.ref.wavelength > 0 * u.m  # Sanity check


def test_link_budget_calculations():
    """Test link budget calculations using test cases."""
    # Test lunar downlink case
    case = load_test_case("lunar_downlink")

    # For system noise temperature, add the magnitudes then compare
    system_temp = (
        case.system_noise_temp.to("K").value
        + case.antenna_noise_temp.to("K").value
    )

    # Verify system noise temperature calculation
    assert system_temp == pytest.approx(
        case.ref.system_noise_temperature.to("K").value, rel=0.01
    )

    # For EIRP calculation, convert tx_power to W and then to dB, then add gain
    import numpy as np
    tx_power_db = 10 * u.dB * np.log10(case.tx_power.to(u.W).value)
    eirp_val = tx_power_db.value + case.tx_antenna_gain.to("dB").value

    # Get the reference value directly without unit conversion
    assert eirp_val == pytest.approx(case.ref.eirp.value, rel=0.01)

    # Test lunar uplink case
    case = load_test_case("lunar_uplink")

    # For system noise temperature, add the magnitudes then compare
    system_temp = (
        case.system_noise_temp.to("K").value
        + case.antenna_noise_temp.to("K").value
    )

    # Verify system noise temperature calculation
    assert system_temp == pytest.approx(
        case.ref.system_noise_temperature.to("K").value, rel=0.01
    )

    # For EIRP calculation, convert tx_power to W and then to dB, then add gain
    tx_power_db = 10 * u.dB * np.log10(case.tx_power.to(u.W).value)
    eirp_val = tx_power_db.value + case.tx_antenna_gain.to("dB").value

    # Get the reference value directly without unit conversion
    assert eirp_val == pytest.approx(case.ref.eirp.value, rel=0.01)


def test_available_test_cases():
    """Test that we can list all available test cases."""
    # Get the list of test cases
    cases = list_test_cases()

    # Verify that our test cases are available
    assert "deep_space" in cases
    assert "leo_downlink" in cases
    assert "leo_uplink" in cases
    assert "lunar_downlink" in cases
    assert "lunar_uplink" in cases

    # Load each test case to make sure they're valid
    for case_name in cases:
        case = load_test_case(case_name)
        assert case.name == case_name
        # Use astropy units for comparisons
        assert case.frequency > 0 * u.GHz
        assert case.distance > 0 * u.m
