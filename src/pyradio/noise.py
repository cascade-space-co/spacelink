"""
Functions for calculating noise in radio systems.

This module provides utilities for calculating various noise-related parameters
in radio systems, including thermal noise, noise figure, and noise temperature.
"""

import numpy as np
from .conversions import db, db2linear

BOLTZMANN_CONSTANT = 1.380649e-23  # J/K
T0 = 290.0  # Reference temperature in Kelvin


def thermal_noise_power(bandwidth: float, temperature: float = 290.0) -> float:
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
        float: Thermal noise power in watts

    Examples:
        >>> thermal_noise_power(1e6)  # 1 MHz bandwidth at 290K
        4.002e-15
    """
    if bandwidth < 0:
        raise ValueError("Bandwidth cannot be negative")
    return BOLTZMANN_CONSTANT * temperature * bandwidth


def noise_figure_to_temperature(noise_figure_db: float) -> float:
    """
    Convert noise figure in dB to equivalent noise temperature.

    The equivalent noise temperature is given by:
    T = T0 * (F - 1)
    where:
    - T0 is the reference temperature (290K)
    - F is the noise factor (linear scale)

    Args:
        noise_figure_db: Noise figure in dB

    Returns:
        float: Equivalent noise temperature in Kelvin

    Raises:
        ValueError: If noise figure is negative

    Examples:
        >>> noise_figure_to_temperature(3.0)  # 3 dB noise figure
        288.6
    """
    if noise_figure_db < 0:
        raise ValueError("Noise figure cannot be negative")

    noise_factor = db2linear(noise_figure_db)
    return T0 * (noise_factor - 1)


def temperature_to_noise_figure(temperature: float) -> float:
    """
    Convert noise temperature to noise figure in dB.

    The noise figure is given by:
    F = 1 + (T/T0)
    where:
    - T is the noise temperature
    - T0 is the reference temperature (290K)

    Args:
        temperature: Noise temperature in Kelvin

    Returns:
        float: Noise figure in dB

    Raises:
        ValueError: If temperature is negative

    Examples:
        >>> temperature_to_noise_figure(288.6)  # ~3 dB noise figure
        3.0
    """
    if temperature < 0:
        raise ValueError("Temperature cannot be negative")

    noise_factor = 1 + (temperature / T0)
    return db(noise_factor)
