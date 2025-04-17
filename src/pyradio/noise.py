"""
Functions for calculating noise in radio systems.

This module provides utilities for calculating various noise-related parameters
in radio systems, including thermal noise, noise figure, and noise temperature.
"""
from pint import Quantity
from pyradio.units import Hz, Q_

# Define constants with units
BOLTZMANN = Q_(1.380649e-23, 'J/K')
T0 = Q_(290.0, 'K')


def power(bandwidth: Quantity, temperature: Quantity = T0) -> Quantity:
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
        >>> import math
        >>> math.isclose(thermal_noise_power(1e6), 4.003e-15, rel_tol=1e-3)
        True
    """
    if bandwidth < 0:
        raise ValueError("Bandwidth cannot be negative")

    return BOLTZMANN * temperature * bandwidth.to(Hz)
