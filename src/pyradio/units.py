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
SPEED_OF_LIGHT = Q_(299792458.0, "m/s")


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


def db_to_lin(value: float) -> float:
    return float(np.pow(10, value / 10.0))


def return_loss_to_vswr(return_loss: float) -> float:
    """
    Convert a return loss in dB to VSWR
    """
    if return_loss < 0:
        raise ValueError(f"return loss must be >= 0 ({return_loss}).")
    if return_loss == float("inf"):
        return 1.0
    return (1 + np.pow(10, -return_loss / 20)) / (1 - np.pow(10, -return_loss / 20))


def vswr_to_return_loss(vswr: float) -> float:
    if vswr <= 1.0:
        raise ValueError(f"VSWR must be > 1 ({vswr}).")
    if np.isclose(vswr, 1.0):
        return float("inf")
    gamma = (vswr - 1) / (vswr + 1)
    return -2 * db(gamma)


# YAML support for Pint Quantity
import yaml


def _quantity_representer(dumper, data: Quantity):
    """YAML representer for Pint Quantity objects."""
    return dumper.represent_mapping(
        "!Quantity", {"magnitude": data.magnitude, "units": str(data.units)}
    )


def _quantity_constructor(loader, node):
    """YAML constructor for Pint Quantity objects."""
    mapping = loader.construct_mapping(node)
    return Q_(mapping["magnitude"], mapping["units"])


# Register representer and constructor with PyYAML safe dumper/loader
yaml.SafeDumper.add_representer(Quantity, _quantity_representer)
yaml.SafeLoader.add_constructor("!Quantity", _quantity_constructor)
