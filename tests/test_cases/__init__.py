"""
Test case definitions for PyRadio.

This module provides utilities for loading and working with predefined test cases
that are shared across different test modules. Each test case contains input parameters
and pre-computed reference values for verification.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

@dataclass
class RefValues:
    """
    A dataclass representing pre-computed reference values for radio link calculations.
    
    This class stores all the reference values for verification of calculations.
    All field names should match the corresponding calculation outputs from the library.
    
    Attributes:
        wavelength_m: Wavelength in meters
        free_space_path_loss_db: Free space path loss in dB
        path_loss_db: Total path loss (including atmospheric and pointing losses) in dB
        eirp_dbw: Effective Isotropic Radiated Power in dBW
        received_power_dbw: Received power in dBW
        system_noise_temperature_k: System noise temperature in Kelvin
        noise_power_dbw: Noise power in dBW
        carrier_to_noise_ratio_db: Carrier to noise ratio in dB
        ebno_db: Energy per bit to noise power spectral density ratio in dB
        link_margin_db: Link margin in dB
    """
    # Physical parameters
    wavelength_m: float = 0.0
    
    # Antenna parameters
    tx_dish_3db_beamwidth_deg: Optional[float] = None
    rx_dish_3db_beamwidth_deg: Optional[float] = None
    
    # Path loss parameters
    free_space_path_loss_db: float = 0.0
    path_loss_db: float = 0.0
    
    # Link budget parameters
    eirp_dbw: float = 0.0
    received_power_dbw: float = 0.0
    system_noise_temperature_k: float = 0.0
    noise_power_dbw: float = 0.0
    carrier_to_noise_ratio_db: float = 0.0
    ebno_db: float = 0.0
    link_margin_db: float = 0.0
    
    # Additional parameters
    data_rate_bps: Optional[float] = None
    ground_station_g_over_t_db: Optional[float] = None

@dataclass
class RadioTestCase:
    """
    A dataclass representing a reference radio link test case.
    
    This class stores input parameters and pre-computed reference values for 
    verification across different test modules. All values are pre-computed and
    stored, so that they can be used to verify the accuracy of calculations.
    
    Attributes:
        name: Unique identifier for the test case
        description: Human-readable description of the test scenario
        
        # Input parameters
        frequency_ghz: Carrier frequency in GHz
        distance_km: Distance in kilometers
        
        # Bandwidth parameters (one of these should be specified)
        bandwidth_mhz: Signal bandwidth in MHz (optional)
        bandwidth_khz: Signal bandwidth in kHz (optional)
        bandwidth_gbps: Signal bandwidth in Gbps (optional)
        
        # Antenna parameters
        tx_dish_diameter_m: Transmitter dish diameter in meters (if applicable)
        tx_dish_efficiency: Transmitter dish efficiency (if applicable)
        tx_antenna_gain_db: Transmitter antenna gain in dB
        rx_dish_diameter_m: Receiver dish diameter in meters (if applicable)
        rx_dish_efficiency: Receiver dish efficiency (if applicable)
        rx_antenna_gain_db: Receiver antenna gain in dB
        
        # Link parameters
        tx_power_dbw: Transmitter power in dBW
        system_noise_temp_k: Receiver system noise temperature in Kelvin
        antenna_noise_temp_k: Receiver antenna noise temperature in Kelvin
        
        # Reference values for various calculations
        ref: Reference values object with pre-computed values
    """
    
    # Case metadata
    name: str
    description: str
    
    # Frequency and distance
    frequency_ghz: float
    distance_km: float
    
    # Bandwidth parameters (one of these should be specified)
    bandwidth_mhz: Optional[float] = None
    bandwidth_khz: Optional[float] = None
    bandwidth_gbps: Optional[float] = None
    
    # Frequency parameters (for other bands/links)
    frequency_thz: Optional[float] = None
    uplink_frequency_mhz: Optional[float] = None
    
    # Antenna parameters
    tx_dish_diameter_m: Optional[float] = None  
    tx_dish_efficiency: Optional[float] = None
    tx_telescope_diameter_cm: Optional[float] = None
    tx_telescope_efficiency: Optional[float] = None
    tx_antenna_gain_db: Optional[float] = None
    rx_dish_diameter_m: Optional[float] = None
    rx_dish_efficiency: Optional[float] = None
    rx_telescope_diameter_cm: Optional[float] = None
    rx_telescope_efficiency: Optional[float] = None
    rx_antenna_gain_db: Optional[float] = None
    
    # Link parameters
    tx_power_dbw: Optional[float] = None
    uplink_tx_power_dbw: Optional[float] = None
    uplink_bandwidth_khz: Optional[float] = None
    uplink_rx_antenna_gain_db: Optional[float] = None
    system_noise_temp_k: Optional[float] = None
    antenna_noise_temp_k: Optional[float] = None
    
    # Additional losses
    implementation_loss_db: float = 0.0
    polarization_loss_db: float = 0.0
    pointing_loss_db: float = 0.0
    atmospheric_loss_db: float = 0.0
    required_ebno_db: float = 0.0
    
    # Reference values (kept for backward compatibility)
    ref_values: Dict[str, float] = field(default_factory=dict)
    
    # Structured reference values
    ref: RefValues = field(default_factory=RefValues)


def load_test_case(case_name: str) -> RadioTestCase:
    """
    Load a test case from its YAML file.
    
    Args:
        case_name: Name of the test case (without .yaml extension)
        
    Returns:
        RadioTestCase object with the test case parameters and reference values
        
    Raises:
        FileNotFoundError: If the test case file doesn't exist
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, f"{case_name}.yaml")
    
    with open(file_path, 'r') as f:
        case_data = yaml.safe_load(f)
    
    # Extract ref_values to create a RefValues object
    ref_values_dict = case_data.get('ref_values', {})
    
    # Create a RefValues object
    ref_values = RefValues()
    for key, value in ref_values_dict.items():
        if hasattr(ref_values, key):
            setattr(ref_values, key, value)
    
    # Add the RefValues object to the case data
    case_data['ref'] = ref_values
    
    # Create and return the RadioTestCase
    return RadioTestCase(**case_data)


def list_test_cases() -> List[str]:
    """
    List all available test cases.
    
    Returns:
        List of test case names (without .yaml extension)
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    files = [f for f in os.listdir(base_dir) if f.endswith('.yaml')]
    return [os.path.splitext(f)[0] for f in files]