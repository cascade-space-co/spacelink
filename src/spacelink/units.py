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
from inspect import signature, Parameter
from typing import get_type_hints, get_args, Annotated
import astropy.units as u
import astropy.constants as constants
from astropy.units import Quantity
from typing import Annotated

import numpy as np
import yaml

# Define dBW if missing
if not hasattr(u, 'dBW'):
    u.dBW = u.dB(u.W)
# Define dBW if missing
if not hasattr(u, 'dBm'):
    u.dBm = u.dB(u.mW)
# Define dimensionless unit if missing
if not hasattr(u, 'dimensionless'):
    u.dimensionless = u.dimensionless_unscaled

# Add dB to linear equivalencies for unit conversion
db_equivalencies = [(u.dB, u.dimensionless_unscaled, 
                    lambda x: 10**(x/10), 
                    lambda x: 10 * np.log10(x))]


Decibels = Annotated[Quantity, u.dB]
DecibelWatts = Annotated[Quantity, u.dB(u.W)]
DecibelMilliwatts = Annotated[Quantity, u.dB(u.mW)]
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
            if hint and getattr(hint, '__origin__', None) is Annotated:
                quantity_type, unit = get_args(hint)
                if isinstance(value, Quantity):
                    # Convert to expected unit
                    bound.arguments[name] = value.to(unit)
                else:
                    # Missing unit - check if this is a numeric type
                    if np.isscalar(value) and not isinstance(value, str):
                        param_desc = f"'{name}' in function '{func.__name__}'"
                        expected_type = f"'{unit}'"
                        raise TypeError(
                            f"Parameter {param_desc} expected to be a Quantity with unit {expected_type}, "
                            f"but got a numeric value without units. Try adding '* u.{unit}' to your value."
                        )
                    # Convert raw numeric value to Quantity
                    bound.arguments[name] = quantity_type(value, unit)

        try:
            return func(*bound.args, **bound.kwargs)
        except AttributeError as e:
            if "'numpy.float64' object has no attribute 'to_value'" in str(e):
                # Common error when forgetting to add units to computed values
                func_name = func.__name__
                raise TypeError(
                    f"In function '{func_name}': A numeric value is missing units. "
                    f"You might have forgotten to add '* u.dimensionless' to a calculation result. "
                    f"Original error: {str(e)}"
                ) from None
            raise

    return wrapper

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
    if factor == 10:  # Power ratio
        linear_value = np.power(10, x.value / factor)
    else:  # Field ratio (factor=20)
        linear_value = np.power(10, x.value / factor)
    return linear_value * u.dimensionless

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
    if vswr <= 1.0:
        raise ValueError(f"VSWR must be > 1 ({vswr}).")
    if np.isclose(vswr, 1.0):
        return float("inf")
    gamma = ((vswr - 1) / (vswr + 1))
    return -to_dB(gamma, factor=20)


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

