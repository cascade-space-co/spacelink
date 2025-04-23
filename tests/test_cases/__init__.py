"""
Test case definitions for SpaceLink.

This module provides utilities for loading and working with predefined test cases
that are shared across different test modules. Each test case contains input parameters
and pre-computed reference values for verification.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from astropy.units import Quantity
import astropy.units as u

# Add custom units if needed
if not hasattr(u, 'bps'):
    u.bps = u.def_unit('bps', doc="Bits per second")
    
# Define dBW if not already defined
if not hasattr(u, 'dBW'):
    u.dBW = u.def_unit('dBW', u.dB(u.W))

# Register YAML constructor for Quantity objects
def quantity_constructor(loader, node):
    """Constructor for !Quantity tags in YAML files."""
    mapping = loader.construct_mapping(node)
    value = mapping.get('value')
    
    # Check for different key names that could contain the unit
    unit_str = mapping.get('unit')
    if unit_str is None:
        unit_str = mapping.get('units')
    
    if unit_str is None:
        raise ValueError("Quantity must have 'unit' or 'units' key")
        
    # Handle special cases
    if unit_str == 'linear':
        return float(value) * u.dimensionless_unscaled
    elif unit_str == 'dB/K':
        return float(value) * u.dB / u.K
    elif unit_str == 'dBW':
        # Handle dBW unit differently since u.dB(u.W) syntax may not be supported in some versions
        return float(value) * u.dBW
    else:
        return float(value) * getattr(u, unit_str)

yaml.SafeLoader.add_constructor('!Quantity', quantity_constructor)


@dataclass
class RefValues:
    """
    A dataclass representing pre-computed reference values for radio link calculations.

    This class stores all the reference values for verification of calculations.
    All field names should match the corresponding calculation outputs from the library.

    Attributes:
        wavelength: Wavelength as a Quantity
        free_space_path_loss: Free space path loss as a Quantity
        eirp: Effective Isotropic Radiated Power as a Quantity
        received_power: Received power as a Quantity
        system_noise_temperature: System noise temperature as a Quantity
        noise_power: Noise power as a Quantity
        carrier_to_noise_ratio: Carrier to noise ratio as a Quantity
        ebno: Energy per bit to noise power spectral density ratio as a Quantity
        link_margin: Link margin as a Quantity
    """

    # Physical parameters
    wavelength: Optional[Quantity] = None

    # Antenna parameters
    tx_dish_3db_beamwidth: Optional[Quantity] = None
    rx_dish_3db_beamwidth: Optional[Quantity] = None

    # Path loss parameters
    free_space_path_loss: Optional[Quantity] = None

    # Link budget parameters
    eirp: Optional[Quantity] = None
    received_power: Optional[Quantity] = None
    system_noise_temperature: Optional[Quantity] = None
    noise_power: Optional[Quantity] = None
    carrier_to_noise_ratio: Optional[Quantity] = None
    ebno: Optional[Quantity] = None
    link_margin: Optional[Quantity] = None

    # Additional parameters
    data_rate: Optional[Quantity] = None
    ground_station_g_over_t: Optional[Quantity] = None


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
        frequency: Carrier frequency as a Quantity
        distance: Distance as a Quantity

        # Bandwidth parameters
        bandwidth: Signal bandwidth as a Quantity

        # Antenna parameters
        tx_dish_diameter: Transmitter dish diameter as a Quantity (if applicable)
        tx_dish_efficiency: Transmitter dish efficiency as a Quantity (if applicable)
        tx_antenna_gain: Transmitter antenna gain as a Quantity
        rx_dish_diameter: Receiver dish diameter as a Quantity (if applicable)
        rx_dish_efficiency: Receiver dish efficiency as a Quantity (if applicable)
        rx_antenna_gain: Receiver antenna gain as a Quantity

        # Link parameters
        tx_power: Transmitter power as a Quantity
        system_noise_temp: Receiver system noise temperature as a Quantity
        antenna_noise_temp: Receiver antenna noise temperature as a Quantity

        # Reference values for various calculations
        ref: Reference values object with pre-computed values
    """

    # Case metadata
    name: str
    description: str

    # Frequency and distance
    frequency: Quantity
    distance: Quantity

    # Bandwidth parameters
    bandwidth: Quantity

    # Antenna parameters
    tx_dish_diameter: Optional[Quantity] = None
    tx_dish_efficiency: Optional[Quantity] = None
    tx_antenna_gain: Optional[Quantity] = None
    tx_antenna_axial_ratio: Optional[Quantity] = None

    rx_dish_diameter: Optional[Quantity] = None
    rx_dish_efficiency: Optional[Quantity] = None
    rx_antenna_gain: Optional[Quantity] = None
    rx_antenna_axial_ratio: Optional[Quantity] = None

    # Link parameters
    tx_power: Optional[Quantity] = None
    system_noise_temp: Optional[Quantity] = None
    antenna_noise_temp: Optional[Quantity] = None

    # Additional losses
    implementation_loss: Optional[Quantity] = None
    polarization_loss: Optional[Quantity] = None
    required_ebno: Optional[Quantity] = None

    # Reference values
    ref_values: Dict[str, Any] = field(default_factory=dict)

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

    with open(file_path, "r") as f:
        case_data = yaml.safe_load(f)

    # Convert string values with units to Pint quantities
    processed_data = {}
    for key, value in case_data.items():
        # Skip ref_values, will process separately
        if key == "ref_values":
            continue

        # Store name and description as is
        if key == "name" or key == "description":
            processed_data[key] = value
            continue

        # Convert strings to quantities or handle structured format
        if isinstance(value, str):
            try:
                # Parse string like "5 dB" into a Quantity
                val, unit_str = value.split(" ", 1)
                # Handle special cases
                if unit_str == 'linear':
                    qty = float(val) * u.dimensionless_unscaled
                elif unit_str == 'dB/K':
                    qty = float(val) * u.dB / u.K
                elif unit_str == 'dBW':
                    qty = float(val) * u.dB(u.W)
                else:
                    qty = float(val) * getattr(u, unit_str)
                processed_data[key] = qty
            except (ValueError, TypeError):
                processed_data[key] = value
        elif isinstance(value, dict) and "value" in value and "unit" in value:
            # Handle the new structured format
            try:
                unit_str = value["unit"]
                qty_value = value["value"]
                # Handle special cases
                if unit_str == 'linear':
                    qty = float(qty_value) * u.dimensionless_unscaled
                elif unit_str == 'dB/K':
                    qty = float(qty_value) * u.dB / u.K
                else:
                    qty = float(qty_value) * getattr(u, unit_str)
                processed_data[key] = qty
            except (ValueError, TypeError):
                processed_data[key] = value
        else:
            processed_data[key] = value

    # Extract ref_values to create a RefValues object
    ref_values_dict = case_data.get("ref_values", {})

    # Create a RefValues object and populate with Pint quantities
    ref_values = RefValues()
    for key, value in ref_values_dict.items():
        if hasattr(ref_values, key):
            if isinstance(value, str):
                try:
                    # Parse string like "5 dB" into a Quantity
                    val, unit_str = value.split(" ", 1)
                    # Handle special cases
                    if unit_str == 'linear':
                        qty = float(val) * u.dimensionless_unscaled
                    elif unit_str == 'dB/K':
                        qty = float(val) * u.dB / u.K
                    elif unit_str == 'dBW':
                        qty = float(val) * u.dBW
                    else:
                        qty = float(val) * getattr(u, unit_str)
                    setattr(ref_values, key, qty)
                except (ValueError, TypeError):
                    setattr(ref_values, key, value)
            elif isinstance(value, dict) and "value" in value and "unit" in value:
                # Handle the new structured format
                try:
                    unit_str = value["unit"]
                    qty_value = value["value"]
                    # Handle special cases
                    if unit_str == 'linear':
                        qty = float(qty_value) * u.dimensionless_unscaled
                    elif unit_str == 'dB/K':
                        qty = float(qty_value) * u.dB / u.K
                    elif unit_str == 'dBW':
                        qty = float(qty_value) * u.dBW
                    else:
                        qty = float(qty_value) * getattr(u, unit_str)
                    setattr(ref_values, key, qty)
                except (ValueError, TypeError):
                    setattr(ref_values, key, value)
            else:
                setattr(ref_values, key, value)

    # Add the RefValues object to the processed data
    processed_data["ref"] = ref_values
    processed_data["ref_values"] = ref_values_dict  # for backward compatibility

    # Create and return the RadioTestCase
    return RadioTestCase(**processed_data)


def list_test_cases() -> List[str]:
    """
    List all available test cases.

    Returns:
        List of test case names (without .yaml extension)
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Exclude cascade fixture files from radio test cases
    files = [
        f
        for f in os.listdir(base_dir)
        if f.endswith(".yaml") and not f.startswith("cascade_")
    ]
    return [os.path.splitext(f)[0] for f in files]
