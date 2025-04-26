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

from functools import wraps
from inspect import signature
from typing import get_type_hints, get_args, Annotated
import astropy.units as u
import astropy.constants as constants
from astropy.units import Quantity
import yaml

import numpy as np

# Define dBW if missing
if not hasattr(u, "dBW"):
    u.dBW = u.dB(u.W)
# Define dBW if missing
if not hasattr(u, "dBm"):
    u.dBm = u.dB(u.mW)
# Define dimensionless unit if missing
if not hasattr(u, "dimensionless"):
    u.dimensionless = u.dimensionless_unscaled

# Add dB to linear equivalencies for unit conversion
db_equivalencies = [
    (
        u.dB,
        u.dimensionless_unscaled,
        lambda x: 10 ** (x / 10),
        lambda x: 10 * np.log10(x),
    )
]


Decibels = Annotated[Quantity, u.dB]
DecibelWatts = Annotated[Quantity, u.dB(u.W)]
DecibelMilliwatts = Annotated[Quantity, u.dB(u.mW)]
Watts = Annotated[Quantity, u.W]
Frequency = Annotated[Quantity, u.Hz]
Wavelength = Annotated[Quantity, u.m]
Dimensionless = Annotated[Quantity, u.dimensionless_unscaled]
Distance = Annotated[Quantity, u.m]
Temperature = Annotated[Quantity, u.K]
Length = Annotated[Quantity, u.m]


def enforce_units(func):
    sig = signature(func)
    hints = get_type_hints(func, include_extras=True)

    @wraps(func)
    def wrapper(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        for name, value in bound.arguments.items():
            hint = hints.get(name)
            # Check if hint is Annotated
            if hint and getattr(hint, "__origin__", None) is Annotated:
                _, unit = get_args(hint) # Use _ for quantity_type if not needed

                if isinstance(value, Quantity):
                    # Convert to expected unit
                    try:
                        if unit.is_equivalent(u.K):
                            converted_value = value.to(unit, equivalencies=u.temperature())
                        else:
                            converted_value = value.to(unit)
                    except u.UnitConversionError as e:
                         raise u.UnitConversionError(f"Parameter '{name}' requires unit compatible with {unit}, but got {value.unit}. Original error: {e}") from e

                    # Unit conversion successful
                    bound.arguments[name] = converted_value

                    # --- Value Checks Removed - Handled by separate decorators ---
                    # if unit == u.K and converted_value < 0 * unit:
                    #    raise ValueError(f"{name} must be non-negative")
                    # if unit == u.Hz and converted_value <= 0 * unit:
                    #    raise ValueError(f"{name} must be positive")
                    # if unit == u.m and name == "distance" and converted_value <= 0 * unit:
                    #    raise ValueError(f"{name} must be positive")

                else:
                    # Handle non-Quantity inputs
                    raise TypeError(f"Parameter '{name}' must be provided as an astropy Quantity, not a raw number.")
            # else: No Annotated hint found
            #    pass

        try:
            return func(*bound.args, **bound.kwargs)
        except AttributeError as e:
            if "'numpy.float64' object has no attribute 'to_value'" in str(e):
                # Common error when forgetting to add units to computed values
                func_name = func.__name__
                raise TypeError(
                    f"In function '{func_name}': A numeric value is missing units. "
                    f"You might have forgotten to add '* u.dimensionless' to a calculation "
                    f"result. Original error: {str(e)}"
                ) from None
            raise

    return wrapper


# DO NOT MODIFY
@enforce_units
def wavelength(frequency: Frequency) -> Wavelength:
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
    return constants.c / frequency.to(u.Hz)


# DO NOT MODIFY
@enforce_units
def frequency(wavelength: Wavelength) -> Frequency:
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
    return constants.c / wavelength.to(u.m)


# DO NOT MODIFY
@enforce_units
def to_dB(x: Dimensionless, *, factor=10) -> Decibels:
    """
    Convert a dimensionless quantity to decibels.

    Args:
        x: A dimensionless Quantity (e.g., power ratio).
        factor: 10 for power, 20 for field (voltage, current, etc.)

    Returns:
        Quantity in decibels (unit = u.dB)
    """
    return factor * u.dB * np.log10(x.to_value(u.dimensionless_unscaled))


# DO NOT MODIFY
@enforce_units
def to_linear(x: Decibels, *, factor: float = 10) -> Dimensionless:
    """
    Convert decibels to a linear (dimensionless) ratio.

    Args:
        x: A quantity in decibels.
        factor: 10 for power quantities, 20 for field quantities.

    Returns:
        A dimensionless quantity (e.g., gain or ratio).
    """
    linear_value = np.power(10, x.value / factor)
    return linear_value * u.dimensionless


# DO NOT MODIFY
@enforce_units
def return_loss_to_vswr(return_loss: Decibels) -> Dimensionless:
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
    if return_loss.value < 0:
        raise ValueError(f"return loss must be >= 0 ({return_loss}).")
    if return_loss.value == float("inf"):
        return 1.0 * u.dimensionless
    gamma = to_linear(-return_loss, factor=20)
    return ((1 + gamma) / (1 - gamma)) * u.dimensionless


# DO NOT MODIFY
@enforce_units
def vswr_to_return_loss(vswr: Dimensionless) -> Decibels:
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
    if vswr < 1.0:
        raise ValueError(f"VSWR must be >= 1 ({vswr}).")
    if np.isclose(vswr.to_value(u.dimensionless), 1.0):
        return float("inf") * u.dB
    gamma = (vswr - 1) / (vswr + 1)
    return -to_dB(gamma, factor=20)


# DO NOT MODIFY
@enforce_units
def mismatch_loss(return_loss: Decibels) -> Decibels:
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
    # Note that we want |Γ|² so we use factor=10 instead of factor=20
    gamma_2 = to_linear(-return_loss, factor=10)
    # Power loss is 1 - |Γ|²
    return -to_dB(1 - gamma_2)


# Register YAML constructor for Quantity objects
def quantity_constructor(loader, node):
    """Constructor for !Quantity tags in YAML files."""
    mapping = loader.construct_mapping(node)
    value = mapping.get("value")

    # Check for different key names that could contain the unit
    unit_str = mapping.get("unit")
    if unit_str is None:
        unit_str = mapping.get("units")

    if unit_str is None:
        raise ValueError("Quantity must have 'unit' or 'units' key")

    # Handle special cases
    if unit_str == "linear":
        return float(value) * u.dimensionless_unscaled
    elif unit_str == "dB/K":
        return float(value) * u.dB / u.K
    elif unit_str == "dBW":
        # Handle dBW unit differently since u.dB(u.W) syntax may not be supported in some versions
        return float(value) * u.dBW
    else:
        return float(value) * getattr(u, unit_str)


# Register the constructor with SafeLoader
yaml.SafeLoader.add_constructor("!Quantity", quantity_constructor)
