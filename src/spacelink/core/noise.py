"""
Functions for calculating noise in radio systems.

Includes functions for noise power, temperature, figure, and factor conversions.
"""

import astropy.units as u

from .units import (
    Decibels,
    Dimensionless,
    Temperature,
    Frequency,
    Power,
    PowerDensity,
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
    Calculate the thermal noise power for a given bandwidth and temperature.

    Parameters
    ----------
    bandwidth : Quantity
        Bandwidth in Hz
    temperature : Quantity, optional
        Noise temperature in Kelvin (default: 290 K)

    Returns
    -------
    Quantity
        Noise power in Watts
    """
    # Check for negative bandwidth remains
    if bandwidth < 0 * u.Hz:
        raise ValueError("Bandwidth cannot be negative")

    result = BOLTZMANN * temperature.to(u.K) * bandwidth.to(u.Hz)
    return result.to(u.W)


@enforce_units
def noise_power_density(temperature: Temperature) -> PowerDensity:
    r"""
    Calculate the noise power density for a given temperature.

    Parameters
    ----------
    temperature : Quantity
        Noise temperature in Kelvin

    Returns
    -------
    Quantity
        Noise power density in Watts per Hertz
    """
    result = BOLTZMANN * temperature.to(u.K)
    return result.to(u.W / u.Hz)


@enforce_units
def temperature_to_noise_factor(temperature: Temperature) -> Dimensionless:
    r"""
    Convert noise temperature to noise factor (linear).

    Parameters
    ----------
    temperature : Quantity
        Noise temperature in Kelvin

    Returns
    -------
    Quantity
        Noise factor (dimensionless, linear)
    """
    return (1.0 + (temperature.to(u.K) / T0)).to(u.dimensionless)


@enforce_units
def noise_factor_to_temperature(noise_factor: Dimensionless) -> Temperature:
    r"""
    Convert noise factor (linear) to noise temperature.

    Parameters
    ----------
    noise_factor : Quantity
        Noise factor (dimensionless, linear)

    Returns
    -------
    Quantity
        Noise temperature in Kelvin
    """
    if noise_factor < 1:
        raise ValueError(f"noise_factor must be >= 1 ({noise_factor})")
    result = (noise_factor - 1.0) * T0
    return result.to(u.K)


@enforce_units
def noise_figure_to_temperature(noise_figure: Decibels) -> Temperature:
    r"""
    Convert noise figure (in dB) to noise temperature (in Kelvin).

    Parameters
    ----------
    noise_figure : Quantity
        Noise figure in dB

    Returns
    -------
    Quantity
        Noise temperature in Kelvin
    """
    factor = to_linear(noise_figure)
    return noise_factor_to_temperature(factor)


@enforce_units
def temperature_to_noise_figure(temperature: Temperature) -> Decibels:
    r"""
    Convert noise temperature in Kelvin to noise figure in dB.

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
