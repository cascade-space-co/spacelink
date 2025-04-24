"""
Functions for calculating noise in radio systems.

This module provides utilities for calculating various noise-related parameters
in radio systems, including thermal noise, noise figure, and noise temperature.
"""

import astropy.units as u
from astropy.units import Quantity
from typing import List

from spacelink.units import (
    Decibels,
    Dimensionless,
    Temperature,
    Frequency,
    enforce_units,
    to_linear,
    to_dB
)


# Define constants with units
BOLTZMANN = 1.380649e-23 * u.J / u.K
T0 = 290.0 * u.K


@enforce_units
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
    noise_factors: List[Dimensionless], gains_lin: List[Dimensionless]
) -> Dimensionless:
    """
    Calculate total cascaded noise factor (linear) using Friis formula.

    Args:
        noise_factors: List of noise factors (linear, dimensionless) for each stage.
        gains_lin: List of linear gains (dimensionless) for each stage.

    Returns:
        Total noise factor (linear, dimensionless) as a Quantity.
    """
    if not noise_factors:
        raise ValueError("Cannot calculate cascaded noise factor for empty stages.")
    if len(noise_factors) != len(gains_lin):
        raise ValueError("noise_factors and gains_lin must have the same length.")
    total_nf = noise_factors[0]
    cum_gain = gains_lin[0]
    for nf, gain in zip(noise_factors[1:], gains_lin[1:]):
        total_nf = total_nf + (nf - 1.0) / cum_gain
        cum_gain = cum_gain * gain
    return total_nf


@enforce_units
def cascaded_noise_figure(
    noise_factors: List[Dimensionless], gains_lin: List[Dimensionless]
) -> Decibels:
    """
    Calculate total cascaded noise figure in dB.

    Args:
        noise_factors: List of noise factors (linear, dimensionless) for each stage.
        gains_lin: List of linear gains (dimensionless) for each stage.

    Returns:
        Total noise figure (dB) as a Quantity.
    """
    total_nf = cascaded_noise_factor(noise_factors, gains_lin)
    return to_dB(total_nf)


@enforce_units
def cascaded_noise_temperature(
    noise_temps: List[Temperature], gains_lin: List[Dimensionless]
) -> Temperature:
    """
    Calculate total cascaded noise temperature in Kelvin.

    Args:
        noise_temps: List of noise temperatures (Kelvin) for each stage.
        gains_lin: List of linear gains (dimensionless) for each stage.

    Returns:
        Total noise temperature (Kelvin) as a Quantity.
    """
    # Compute cascaded noise temperature via total noise factor
    if not noise_temps:
        raise ValueError(
            "Cannot calculate cascaded noise temperature for empty stages."
        )
    if len(noise_temps) != len(gains_lin):
        raise ValueError("noise_temps and gains_lin must have the same length.")
    # Convert noise temperatures to noise factors: F = 1 + T/T0
    noise_factors = [temperature_to_noise_factor(temp) for temp in noise_temps]
    # Compute total noise factor
    total_nf = cascaded_noise_factor(noise_factors, gains_lin)
    # Convert back to noise temperature: T = (F - 1) * T0
    return noise_factor_to_temperature(total_nf)
