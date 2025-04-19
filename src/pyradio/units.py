"""
Units and constants for radio communications calculations.
"""

from pint import UnitRegistry, Quantity
import numpy as np
# Create a unit registry
# Autoconvert offset to base units is important for logarithmic operations
ureg = UnitRegistry(autoconvert_offset_to_baseunit=True)

# Define a Quantity type alias for better type hints
Q_ = ureg.Quantity

# Define frequency units
Hz = ureg.Hz
kHz = ureg.kHz
MHz = ureg.MHz
GHz = ureg.GHz

# Define power units
W = ureg.W
mW = ureg.mW

# Define logarithmic units
dB = ureg.dB
dBW = ureg.dBW
dBm = ureg.dBm

# Define distance units
m = ureg.m
km = ureg.km

# Define temperature units
K = ureg.K

# Define dimensionless units
dimensionless = ureg.dimensionless

# Speed of light in m/s
SPEED_OF_LIGHT = Q_(299792458.0, 'm/s')


def wavelength(frequency: Quantity) -> Quantity:
    """
    Convert frequency to wavelength.

    Args:
        frequency: Frequency in Hz

    Returns:
        Wavelength in meters
    """
    return SPEED_OF_LIGHT / frequency.to(Hz)


def frequency(wavelength: Quantity) -> Quantity:
    """
    Convert wavelength to frequency.

    Args:
        wavelength: Wavelength in meters

    Returns:
        Frequency in Hz
    """
    return SPEED_OF_LIGHT / wavelength.to(m)


def db(value: float) -> float:
    """
    Convert a linear value to decibels.

    Args:
        value: Linear value to convert

    Returns:
        Value in decibels
    """
    return float(10.0 * np.log10(value))
