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

    result = BOLTZMANN * temperature * bandwidth.to(u.Hz)
    return result.to(u.W)


@enforce_units
def temperature_to_noise_factor(temperature: Temperature) -> Dimensionless:
    r"""
    """
    return (1.0 + (temperature / T0)).to(u.dimensionless)


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


@enforce_units
def cascaded_noise_factor(
    stages: List[Tuple[Dimensionless, Dimensionless]],
) -> Dimensionless:
    r"""
    Calculate total cascaded noise factor (linear) using Friis formula.

    The cascaded noise factor is calculated using:

    .. math::
        F_{total} = F_1 + \frac{F_2 - 1}{G_1} + \frac{F_3 - 1}{G_1 G_2}
        + \ldots + \frac{F_n - 1}{G_1 G_2 \ldots G_{n-1}}

    where:

    * :math:`F_i` is the noise factor (linear) of the i-th stage
    * :math:`G_i` is the gain (linear) of the i-th stage

    Parameters
    ----------
    stages : List[Tuple[Dimensionless, Dimensionless]]
        List of tuples containing (noise_factor, gain) for each stage.
        noise_factor: Noise factor (linear, dimensionless)
        gain: Linear gain (dimensionless)

    Returns
    -------
    Quantity
        Total noise factor (linear, dimensionless)

    Raises
    ------
    ValueError
        If stages list is empty
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
def cascaded_noise_figure(stages: List[Tuple[Decibels, Decibels]]) -> Decibels:
    r"""
    Calculate total cascaded noise figure in dB.

    This function converts the noise figure and gain values from dB to linear,
    calls cascaded_noise_factor, and then converts the result back to dB.

    Parameters
    ----------
    stages : List[Tuple[Decibels, Decibels]]
        List of tuples containing (noise_figure, gain) for each stage.
        noise_figure: Noise figure in dB
        gain: Gain in dB

    Returns
    -------
    Quantity
        Total noise figure in dB

    Raises
    ------
    ValueError
        If stages list is empty
    """
    # Convert dB values to linear
    linear_stages = [(to_linear(nf), to_linear(gain)) for nf, gain in stages]
    return to_dB(cascaded_noise_factor(linear_stages))


@enforce_units
def cascaded_noise_temperature(
    stages: List[Tuple[Temperature, Dimensionless]],
) -> Temperature:
    r"""
    Calculate total cascaded noise temperature in Kelvin.

    This function converts noise temperatures to noise factors, calculates
    the cascaded noise factor, and then converts back to temperature.

    The calculation follows these steps:

    1. Convert each noise temperature to a noise factor:

       .. math::
           F_i = 1 + \frac{T_i}{T_0}

    2. Calculate the cascaded noise factor using Friis' formula
    3. Convert the result back to temperature:

       .. math::
           T_{total} = (F_{total} - 1) \cdot T_0

    Parameters
    ----------
    stages : List[Tuple[Temperature, Dimensionless]]
        List of tuples containing (noise_temperature, gain) for each stage.
        noise_temperature: Noise temperature in Kelvin
        gain: Linear gain (dimensionless)

    Returns
    -------
    Quantity
        Total noise temperature in Kelvin

    Raises
    ------
    ValueError
        If stages list is empty
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
