"""
Units and constants for radio communications calculations.

This module defines a Pint UnitRegistry and provides commonly used units and conversion
functions for radio frequency applications, including:
  - Wavelength and frequency conversions
  - Decibel and linear scale conversions
  - VSWR and return loss calculations
  - Mismatch loss computation
  - YAML serialization support for Pint quantities
"""

from pint import UnitRegistry, Quantity
import numpy as np
import yaml

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
        frequency (pint.Quantity): Frequency quantity (e.g., in Hz).

    Returns:
        pint.Quantity: Wavelength in meters.

    Raises:
        Exception: If the input quantity has incompatible units.

    Example:
        >>> from spacelink.units import wavelength, GHz, m
        >>> wavelength(1 * GHz).to(m)
        <Quantity(0.299792458, 'meter')>
    """
    return SPEED_OF_LIGHT / frequency.to(Hz)


def frequency(wavelength: Quantity) -> Quantity:
    """
    Convert wavelength to frequency.

    Args:
        wavelength (pint.Quantity): Wavelength quantity (e.g., in meters).

    Returns:
        pint.Quantity: Frequency in hertz.

    Raises:
        Exception: If the input quantity has incompatible units.

    Example:
        >>> from spacelink.units import frequency, m, MHz
        >>> frequency(1 * m).to(MHz)
        <Quantity(299.792458, 'megahertz')>
    """
    return SPEED_OF_LIGHT / wavelength.to(m)


def db(value: float) -> float:
    """
    Convert a linear scale value to decibels (10 * log10).

    Args:
        value (float): Linear value to convert; must be positive.

    Returns:
        float: Value in decibels.

    Raises:
        ValueError: If value is not positive.

    Example:
        >>> db(10.0)
        10.0
    """
    # Ensure valid input
    if value <= 0:
        raise ValueError(f"value must be > 0 ({value}).")
    return float(10.0 * np.log10(value))


def db_to_lin(value: float) -> float:
    """
    Convert a decibel value to a linear scale ratio.

    Args:
        value (float): Value in decibels.

    Returns:
        float: Linear scale value.

    Example:
        >>> db_to_lin(20.0)
        100.0
    """
    return float(np.pow(10, value / 10.0))


def return_loss_to_vswr(return_loss: float) -> float:
    """
    Convert a return loss in decibels to voltage standing wave ratio (VSWR).

    Args:
        return_loss (float): Return loss in decibels (>= 0). Use float('inf') for a perfect match.

    Returns:
        float: VSWR (>= 1).

    Raises:
        ValueError: If return_loss is negative.

    Example:
        >>> return_loss_to_vswr(20.0)
        1.2
    """
    if return_loss < 0:
        raise ValueError(f"return loss must be >= 0 ({return_loss}).")
    if return_loss == float("inf"):
        return 1.0
    return (1 + np.pow(10, -return_loss / 20)) / (1 - np.pow(10, -return_loss / 20))


def vswr_to_return_loss(vswr: float) -> float:
    """
    Convert voltage standing wave ratio (VSWR) to return loss in decibels.

    Args:
        vswr (float): VSWR value (> 1). Use 1 for a perfect match (infinite return loss).

    Returns:
        float: Return loss in decibels.

    Raises:
        ValueError: If vswr is less than or equal to 1.

    Example:
        >>> vswr_to_return_loss(1.2)
        20.834...
    """
    if vswr <= 1.0:
        raise ValueError(f"VSWR must be > 1 ({vswr}).")
    if np.isclose(vswr, 1.0):
        return float("inf")
    gamma = (vswr - 1) / (vswr + 1)
    return -2 * db(gamma)


def mismatch_loss(return_loss: float) -> float:
    """
    Compute the mismatch loss due to non-ideal return loss.

    Mismatch loss quantifies power lost from reflections at an interface.

    Args:
        return_loss (float): Return loss in decibels.

    Returns:
        float: Mismatch loss in decibels.

    Example:
        >>> mismatch_loss(9.54)
        0.5115...
    """
    gamma = np.pow(10, -return_loss / 20.0)
    return -db(1 - gamma**2)


# YAML support for Pint Quantity


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
