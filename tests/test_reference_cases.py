"""
Tests that verify the consistency of the reference test cases.

These tests demonstrate how to use the reference test cases for
verifying various functions across the codebase.
"""

import pytest
from pyradio.conversions import ghz, wavelength
from pyradio.antenna import dish_gain, dish_3db_beamwidth
from pyradio.path import free_space_path_loss_db
from test_cases import load_test_case, list_test_cases

def test_wavelength_reference():
    """Test wavelength calculation using reference test cases."""
    # Load the deep space case
    case = load_test_case("deep_space")
    
    # Calculate wavelength using the wavelength function
    calc_wavelength = wavelength(ghz(case.frequency_ghz))
    
    # Verify against reference value
    assert calc_wavelength == pytest.approx(case.ref.wavelength_m, rel=0.001)

    # Load the lunar downlink case
    case = load_test_case("lunar_downlink")
    
    # Calculate wavelength using the wavelength function
    calc_wavelength = wavelength(ghz(case.frequency_ghz))
    
    # Verify against reference value
    assert calc_wavelength == pytest.approx(case.ref.wavelength_m, rel=0.001)

def test_dish_gain_reference():
    """Test dish gain calculation using reference test case."""
    # Load deep space case with dish antennas
    case = load_test_case("deep_space")
    
    # Calculate transmitter dish gain
    calc_gain = dish_gain(
        diameter=case.tx_dish_diameter_m,
        frequency=ghz(case.frequency_ghz),
        efficiency=case.tx_dish_efficiency
    )
    
    # Allow for a margin of error in the reference value
    # This is more realistic than requiring exact match
    assert calc_gain == pytest.approx(case.tx_antenna_gain_db, rel=0.05)
    
    # Calculate receiver dish gain
    calc_gain = dish_gain(
        diameter=case.rx_dish_diameter_m,
        frequency=ghz(case.frequency_ghz),
        efficiency=case.rx_dish_efficiency
    )
    
    # Allow for a margin of error in the reference value
    assert calc_gain == pytest.approx(case.rx_antenna_gain_db, rel=0.05)

def test_dish_beamwidth_reference():
    """Test dish beamwidth calculation using reference test case."""
    # Load deep space case with dish antennas
    case = load_test_case("deep_space")
    
    # Calculate transmitter dish beamwidth
    calc_beamwidth = dish_3db_beamwidth(
        diameter=case.tx_dish_diameter_m,
        frequency=ghz(case.frequency_ghz)
    )
    
    # Allow for a margin of error in the reference value
    assert calc_beamwidth == pytest.approx(case.ref.tx_dish_3db_beamwidth_deg, rel=0.05)
    
    # Calculate receiver dish beamwidth
    calc_beamwidth = dish_3db_beamwidth(
        diameter=case.rx_dish_diameter_m,
        frequency=ghz(case.frequency_ghz)
    )
    
    # Allow for a margin of error in the reference value
    assert calc_beamwidth == pytest.approx(case.ref.rx_dish_3db_beamwidth_deg, rel=0.05)

def test_path_loss_reference():
    """Test path loss calculation using reference test cases."""
    # Test with deep space case
    case = load_test_case("deep_space")
    
    # Calculate free space path loss
    calc_fspl = free_space_path_loss_db(
        distance_m=case.distance_km * 1000,  # convert to meters
        frequency_hz=ghz(case.frequency_ghz)
    )
    
    # Verify against reference value with a margin of error
    assert calc_fspl == pytest.approx(case.ref.free_space_path_loss_db, rel=0.01)
    
    # Test with LEO downlink case
    case = load_test_case("leo_downlink")
    
    # Calculate free space path loss
    calc_fspl = free_space_path_loss_db(
        distance_m=case.distance_km * 1000,  # convert to meters
        frequency_hz=ghz(case.frequency_ghz)
    )
    
    # Verify against reference value with a margin of error
    assert calc_fspl == pytest.approx(case.ref.free_space_path_loss_db, rel=0.01)

def test_link_budget_calculations():
    """Test link budget calculations using test cases."""
    # Test lunar downlink case
    case = load_test_case("lunar_downlink")
    
    # Calculate system noise temperature (receiver + antenna)
    system_temp = case.system_noise_temp_k + case.antenna_noise_temp_k
    
    # Verify system noise temperature calculation
    assert system_temp == pytest.approx(case.ref.system_noise_temperature_k, rel=0.01)
    
    # Calculate EIRP (tx_power_dbw + tx_antenna_gain_db)
    eirp = case.tx_power_dbw + case.tx_antenna_gain_db
    
    # Verify EIRP calculation
    assert eirp == pytest.approx(case.ref.eirp_dbw, rel=0.01)
    
    # Test lunar uplink case
    case = load_test_case("lunar_uplink")
    
    # Calculate system noise temperature (receiver + antenna)
    system_temp = case.system_noise_temp_k + case.antenna_noise_temp_k
    
    # Verify system noise temperature calculation
    assert system_temp == pytest.approx(case.ref.system_noise_temperature_k, rel=0.01)
    
    # Calculate EIRP (tx_power_dbw + tx_antenna_gain_db)
    eirp = case.tx_power_dbw + case.tx_antenna_gain_db
    
    # Verify EIRP calculation
    assert eirp == pytest.approx(case.ref.eirp_dbw, rel=0.01)

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
        assert case.frequency_ghz > 0
        assert case.distance_km > 0