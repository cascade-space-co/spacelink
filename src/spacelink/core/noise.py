"""
Functions for calculating noise in radio systems.

Includes functions for noise power, temperature, figure, and factor conversions.
"""

import astropy.units as u
# import astropy.constants as const # Unused
# import numpy as np # Unused
from astropy.units import Quantity
from typing import List, Tuple

from .units import (
    Decibels,
    Dimensionless,
    Temperature,
    Frequency,
    enforce_units,
    to_linear,
    to_dB,
)
# from ..core.validation import check_positive # Removed import


# Define constants with units
BOLTZMANN = 1.380649e-23 * u.J / u.K
T0 = 290.0 * u.K


@enforce_units
# @check_positive("bandwidth") # Removed check
def power(bandwidth: Frequency, temperature: Temperature = T0) -> Quantity:
    """
    Calculate the thermal noise power in a given bandwidth.

    The thermal noise power is given by:
    P = k * T * B
    where:
    - k is Boltzmann's constant (1.380649e-23 J/K)
    - T is the temperature in Kelvin
    - B is the bandwidth in Hz

    Args:
        bandwidth: Bandwidth in Hz
        temperature: Temperature in Kelvin (default: 290K, standard room temperature)

    Returns:
        Thermal noise power in watts

    Examples:
        >>> import math
        >>> math.isclose(power(1e6 * u.Hz).value, 4.003e-15, rel_tol=1e-3)
        True
    """
    # Check for negative bandwidth remains
    if bandwidth < 0 * u.Hz:
       raise ValueError("Bandwidth cannot be negative")

    return BOLTZMANN * temperature * bandwidth.to(u.Hz)


@enforce_units
def temperature_to_noise_factor(temperature: Temperature) -> Dimensionless:
    """
    Convert noise temperature in Kelvin to noise factor (linear).

    The noise factor is given by:
    F = 1 + T/T0
    where:
    - T is the noise temperature in Kelvin
    - T0 is the reference temperature (290K)

    Args:
        temperature: Noise temperature in Kelvin

    Returns:
        Noise factor (linear, dimensionless)
    """
    return 1.0 + (temperature / T0) * u.dimensionless


@enforce_units
def noise_factor_to_temperature(noise_factor: Dimensionless) -> Temperature:
    """
    Convert noise factor (linear) to noise temperature in Kelvin.

    The noise temperature is given by:
    T = (F - 1) * T0
    where:
    - F is the noise factor (linear)
    - T0 is the reference temperature (290K)

    Args:
        noise_factor: Noise factor (linear, dimensionless)

    Returns:
        Noise temperature in Kelvin

    Raises:
        ValueError: If noise factor is less than 1
    """
    if noise_factor < 1:
        raise ValueError(f"noise_factor must be >= 1 ({noise_factor})")
    return (noise_factor - 1.0) * T0


@enforce_units
def noise_figure_to_temperature(noise_figure: Decibels) -> Temperature:
    """
    Convert noise figure in dB to noise temperature in Kelvin.

    Args:
        noise_figure: Noise figure in dB

    Returns:
        Noise temperature in Kelvin
    """
    factor = to_linear(noise_figure)
    return noise_factor_to_temperature(factor)


@enforce_units
def temperature_to_noise_figure(temperature: Temperature) -> Decibels:
    """
    Convert noise temperature in Kelvin to noise figure in dB.

    Args:
        temperature: Noise temperature in Kelvin

    Returns:
        Noise figure in dB
    """
    factor = temperature_to_noise_factor(temperature)
    return to_dB(factor)


@enforce_units
def cascaded_noise_factor(
    stages: List[Tuple[Dimensionless, Dimensionless]]
) -> Dimensionless:
    """
    Calculate total cascaded noise factor (linear) using Friis formula.

    Args:
        stages: List of tuples containing (noise_factor, gain) for each stage.
            noise_factor: Noise factor (linear, dimensionless)
            gain: Linear gain (dimensionless)

    Returns:
        Total noise factor (linear, dimensionless) as a Quantity.
    """
    if not stages:
        raise ValueError("Cannot calculate cascaded noise factor for empty stages.")
    
    total_nf = stages[0][0]  # First stage noise factor
    cum_gain = stages[0][1]  # First stage gain
    
    for nf, gain in stages[1:]:
        total_nf = total_nf + (nf - 1.0) / cum_gain
        cum_gain = cum_gain * gain
    return total_nf


@enforce_units
def cascaded_noise_figure(
    stages: List[Tuple[Decibels, Decibels]]
) -> Decibels:
    """
    Calculate total cascaded noise figure in dB.

    Args:
        stages: List of tuples containing (noise_figure, gain) for each stage.
            noise_figure: Noise figure in dB
            gain: Gain in dB

    Returns:
        Total noise figure (dB) as a Quantity.
    """
    # Convert dB values to linear
    linear_stages = [(to_linear(nf), to_linear(gain)) for nf, gain in stages]
    return to_dB(cascaded_noise_factor(linear_stages))


@enforce_units
def cascaded_noise_temperature(
    stages: List[Tuple[Temperature, Dimensionless]]
) -> Temperature:
    """
    Calculate total cascaded noise temperature in Kelvin.

    Args:
        stages: List of tuples containing (noise_temperature, gain) for each stage.
            noise_temperature: Noise temperature in Kelvin
            gain: Linear gain (dimensionless)

    Returns:
        Total noise temperature (Kelvin) as a Quantity.
    """
    if not stages:
        raise ValueError(
            "Cannot calculate cascaded noise temperature for empty stages."
        )
    
    # Convert noise temperatures to noise factors: F = 1 + T/T0
    noise_factors = [(temperature_to_noise_factor(temp), gain) for temp, gain in stages]
    # Compute total noise factor
    total_nf = cascaded_noise_factor(noise_factors)
    # Convert back to noise temperature: T = (F - 1) * T0
    return noise_factor_to_temperature(total_nf)
