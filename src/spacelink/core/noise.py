"""
Functions for calculating noise in radio systems.

Includes functions for noise power, temperature, figure, and factor conversions.
"""

import astropy.units as u
from typing import List, Tuple

from .units import (
    Decibels,
    Dimensionless,
    Temperature,
    Frequency,
    Power,
    enforce_units,
    to_linear,
    to_dB,
)

# from ..core.validation import check_positive # Removed import


# Define constants with units
BOLTZMANN = 1.380649e-23 * u.J / u.K
T0 = 290.0 * u.K


@enforce_units
def noise_power(bandwidth: Frequency, temperature: Temperature = T0) -> Power:
    r"""
    """
    # Check for negative bandwidth remains
    if bandwidth < 0 * u.Hz:
        raise ValueError("Bandwidth cannot be negative")

    result = BOLTZMANN * temperature.to(u.K) * bandwidth.to(u.Hz)
    return result.to(u.W)


@enforce_units
def temperature_to_noise_factor(temperature: Temperature) -> Dimensionless:
    r"""
    """
    return (1.0 + (temperature.to(u.K) / T0)).to(u.dimensionless)


@enforce_units
def noise_factor_to_temperature(noise_factor: Dimensionless) -> Temperature:
    r"""

    """
    if noise_factor < 1:
        raise ValueError(f"noise_factor must be >= 1 ({noise_factor})")
    result = (noise_factor - 1.0) * T0
    return result.to(u.K)


@enforce_units
def noise_figure_to_temperature(noise_figure: Decibels) -> Temperature:
    r"""
    """
    factor = to_linear(noise_figure)
    return noise_factor_to_temperature(factor)


@enforce_units
def temperature_to_noise_figure(temperature: Temperature) -> Decibels:
    r"""
    Convert noise temperature in Kelvin to noise figure in dB.

    The conversion is done in two steps:

    1. Convert temperature to noise factor (linear):

       .. math::
           F = 1 + \frac{T}{T_0}

    2. Convert noise factor to noise figure (dB):

       .. math::
           NF_{dB} = 10 \log_{10}(F)

    where :math:`T_0` is the reference temperature (290K).

    Parameters
    ----------
    temperature : Quantity
        Noise temperature in Kelvin

    Returns
    -------
    Quantity
        Noise figure in dB
    """
    factor = temperature_to_noise_factor(temperature)
    return to_dB(factor)

