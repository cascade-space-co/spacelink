"""
Units and constants for radio communications calculations.

This module defines an Astropy UnitRegistry and provides commonly used units and conversion
functions for radio frequency applications, including:

- Wavelength and frequency conversions
- Decibel and linear scale conversions
- VSWR and return loss calculations
- Mismatch loss computation
"""

from functools import wraps
from inspect import signature
from typing import get_type_hints, get_args, Annotated
import astropy.units as u
import astropy.constants as constants
from astropy.units import Quantity
import yaml

import numpy as np

if not hasattr(u, "dBHz"):
    u.dBHz = u.dB(u.Hz)
if not hasattr(u, "dBW"):
    u.dBW = u.dB(u.W)
if not hasattr(u, "dBm"):
    u.dBm = u.dB(u.mW)
if not hasattr(u, "dBK"):
    u.dBK = u.dB(u.K)

if not hasattr(u, "dimensionless"):
    u.dimensionless = u.dimensionless_unscaled

# TODO: make unit naming consistent i.e. Watts not Power, Kelvins not Temperature, etc
Decibels = Annotated[Quantity, u.dB]
DecibelWatts = Annotated[Quantity, u.dB(u.W)]
DecibelMilliwatts = Annotated[Quantity, u.dB(u.mW)]
DecibelKelvins = Annotated[Quantity, u.dB(u.K)]
Watts = Annotated[Quantity, u.W]
Power = Annotated[Quantity, u.W]
Frequency = Annotated[Quantity, u.Hz]
Wavelength = Annotated[Quantity, u.m]
Dimensionless = Annotated[Quantity, u.dimensionless_unscaled]
Distance = Annotated[Quantity, u.m]
Temperature = Annotated[Quantity, u.K]
Length = Annotated[Quantity, u.m]
DecibelHertz = Annotated[Quantity, u.dB(u.Hz)]


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
                _, unit = get_args(hint)  # Use _ for quantity_type if not needed

                if isinstance(value, Quantity):
                    # Convert to expected unit
                    try:
                        if unit.is_equivalent(u.K):
                            converted_value = value.to(
                                unit, equivalencies=u.temperature()
                            )
                        else:
                            converted_value = value.to(unit)
                    except u.UnitConversionError as e:
                        raise u.UnitConversionError(
                            f"Parameter '{name}' requires unit compatible with {unit}, "
                            f"but got {value.unit}. Original error: {e}"
                        ) from e

                    # Unit conversion successful
                    bound.arguments[name] = converted_value

                else:
                    # Handle non-Quantity inputs
                    raise TypeError(
                        f"Parameter '{name}' must be provided as an astropy Quantity, "
                        f"not a raw number."
                    )

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
    r"""
    Convert frequency to wavelength.

    The wavelength is calculated using:

    .. math::
        \lambda = \frac{c}{f}

    where:

    * :math:`c` is the speed of light (299,792,458 m/s)
    * :math:`f` is the frequency in Hz

    Parameters
    ----------
    frequency : Quantity
        Frequency quantity (e.g., in Hz)

    Returns
    -------
    Quantity
        Wavelength in meters

    Raises
    ------
    UnitConversionError
        If the input quantity has incompatible units

    Examples
    --------
    >>> from spacelink.units import wavelength, GHz, m
    >>> wavelength(1 * GHz).to(m)
    <Quantity(0.299792458, 'meter')>
    """
    return constants.c / frequency.to(u.Hz)


# DO NOT MODIFY
@enforce_units
def frequency(wavelength: Wavelength) -> Frequency:
    r"""
    Convert wavelength to frequency.

    The frequency is calculated using:

    .. math::
        f = \frac{c}{\lambda}

    where:

    * :math:`c` is the speed of light (299,792,458 m/s)
    * :math:`\lambda` is the wavelength in meters

    Parameters
    ----------
    wavelength : Quantity
        Wavelength quantity (e.g., in meters)

    Returns
    -------
    Quantity
        Frequency in hertz

    Raises
    ------
    UnitConversionError
        If the input quantity has incompatible units

    Examples
    --------
    >>> from spacelink.units import frequency, m, MHz
    >>> frequency(1 * m).to(MHz)
    <Quantity(299.792458, 'megahertz')>
    """
    return constants.c / wavelength.to(u.m)


@enforce_units
def to_dB(x: Quantity, *, factor=10) -> Quantity:
    r"""
    Convert a quantity to decibels, preserving the logarithmic units.

    The conversion is done using:

    .. math::
        X_{{dB}} = factor \cdot \log_{{10}}(x)

    The result will have units of dB(input_unit), e.g. dBW, dBK, dBHz, etc.

    Parameters
    ----------
    x : Quantity
        A Quantity (e.g., in W, K, Hz, etc.)
    factor : int, optional
        10 for power quantities, 20 for field quantities

    Returns
    -------
    Quantity
        Value in decibels with logarithmic units (e.g., dBW, dBK)
    """
    db_value = factor * np.log10(x.value)
    return db_value * u.dB(x.unit)


@enforce_units
def to_dBHz(x: Frequency) -> DecibelHertz:
    r"""
    Convert a decibel quantity to dBHz
    """
    return 10 * np.log10(x.to_value(u.Hz)) * u.dBHz


# DO NOT MODIFY
@enforce_units
def to_linear(x: Decibels, *, factor: float = 10) -> Dimensionless:
    r"""
    Convert decibels to a linear (dimensionless) ratio.

    The conversion is done using:

    .. math::
        x = 10^{\frac{X_{dB}}{factor}}

    where:

    * :math:`X_{dB}` is the value in decibels
    * :math:`factor` is 10 for power quantities, 20 for field quantities

    Parameters
    ----------
    x : Quantity
        A quantity in decibels
    factor : float, optional
        10 for power quantities, 20 for field quantities

    Returns
    -------
    Quantity
        A dimensionless quantity (e.g., gain or ratio)
    """
    linear_value = np.power(10, x.value / factor)
    return linear_value * u.dimensionless


# DO NOT MODIFY
@enforce_units
def return_loss_to_vswr(return_loss: Decibels) -> Dimensionless:
    r"""
    Convert a return loss in decibels to voltage standing wave ratio (VSWR).

    The conversion is done using:

    .. math::
        VSWR = \frac{1 + |\Gamma|}{1 - |\Gamma|}

    where:

    * :math:`|\Gamma|` is the magnitude of the reflection coefficient
    * :math:`|\Gamma| = 10^{-\frac{RL}{20}}`
    * :math:`RL` is the return loss in dB

    Parameters
    ----------
    return_loss : Quantity
        Return loss in decibels (>= 0). Use float('inf') for a perfect match

    Returns
    -------
    Quantity
        VSWR (>= 1)

    Raises
    ------
    ValueError
        If return_loss is negative

    Examples
    --------
    >>> return_loss_to_vswr(20.0 * u.dB)
    <Quantity(1.2, 'dimensionless')>
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
    r"""
    Convert voltage standing wave ratio (VSWR) to return loss in decibels.

    The conversion is done using:

    .. math::
        RL = -20 \log_{10}\left(\frac{VSWR - 1}{VSWR + 1}\right)

    where:

    * :math:`VSWR` is the voltage standing wave ratio
    * :math:`RL` is the return loss in dB

    Parameters
    ----------
    vswr : Quantity
        VSWR value (> 1). Use 1 for a perfect match (infinite return loss)

    Returns
    -------
    Quantity
        Return loss in decibels

    Raises
    ------
    ValueError
        If vswr is less than 1

    Examples
    --------
    >>> vswr_to_return_loss(1.2 * u.dimensionless)
    <Quantity(20.83, 'dB')>
    """
    if vswr < 1.0:
        raise ValueError(f"VSWR must be >= 1 ({vswr}).")
    if np.isclose(vswr.to_value(u.dimensionless), 1.0):
        return float("inf") * u.dB
    gamma = (vswr - 1) / (vswr + 1)
    return safe_negate(to_dB(gamma, factor=20))


# DO NOT MODIFY
@enforce_units
def mismatch_loss(return_loss: Decibels) -> Decibels:
    r"""
    Compute the mismatch loss due to non-ideal return loss.

    Mismatch loss quantifies power lost from reflections at an interface.
    It is calculated using:

    .. math::
        ML = -10 \log_{10}(1 - |\Gamma|^2)

    where:

    * :math:`|\Gamma|` is the magnitude of the reflection coefficient
    * :math:`|\Gamma| = 10^{-\frac{RL}{20}}`
    * :math:`RL` is the return loss in dB

    Parameters
    ----------
    return_loss : Quantity
        Return loss in decibels

    Returns
    -------
    Quantity
        Mismatch loss in decibels

    Examples
    --------
    >>> mismatch_loss(9.54 * u.dB)
    <Quantity(0.51, 'dB')>
    """
    # Note that we want |Γ|² so we use factor=10 instead of factor=20
    gamma_2 = to_linear(-return_loss, factor=10)
    # Power loss is 1 - |Γ|²
    return safe_negate(to_dB(1 - gamma_2))


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

def safe_negate(quantity):
    """
    Safely negate a dB or function unit quantity, preserving the unit.
    Astropy does not allow direct negation of function units (like dB).
    """
    return (-1 * quantity.value) * quantity.unit
