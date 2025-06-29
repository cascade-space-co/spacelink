r"""
Polarization Loss
-----------------

The polarization loss between two antennas with given axial ratios is
calculated using the standard formula for polarization mismatch:

.. math::
   \text{PLF} = \frac{1}{2} +\
   \frac{1}{2} \frac{4 \gamma_1 \gamma_2 -\
   (1-\gamma_1^2)(1-\gamma_2^2)}{(1+\gamma_1^2)(1+\gamma_2^2)}

where:

* :math:`\gamma_1` and :math:`\gamma_2` are the voltage axial ratios (linear, not dB)
* PLF is the polarization loss factor (linear)

The polarization loss in dB is then:

.. math::
   L_{\text{pol}} = -10 \log_{10}(\text{PLF})

For circular polarization, the axial ratio is 0 dB, and for linear polarization,
it is >40 dB.

Dish Gain
---------

The gain of a parabolic dish antenna is given by:

.. math::
   G = \eta \left(\frac{\pi D}{\lambda}\right)^2

where:

* :math:`\eta` is the efficiency factor (typically 0.55 to 0.70)
* :math:`D` is the diameter of the dish
* :math:`\lambda` is the wavelength
"""

import astropy.units as u
import numpy as np

from .units import (
    Decibels,
    Dimensionless,
    Frequency,
    Length,
    wavelength,
    enforce_units,
    to_dB,
    to_linear,
    safe_negate,
)


@enforce_units
def polarization_loss(ar1: Decibels, ar2: Decibels) -> Decibels:
    r"""
    Calculate the polarization loss in dB between two antennas with given axial ratios.

    Parameters
    ----------
    ar1 : Decibels
        First antenna axial ratio in dB (amplitude ratio)
    ar2 : Decibels
        Second antenna axial ratio in dB (amplitude ratio)

    Returns
    -------
    Decibels
        Polarization loss in dB (positive value)
    """
    # Polarization mismatch angle is omitted (assumed to be 90 degrees)
    # https://www.microwaves101.com/encyclopedias/polarization-mismatch-between-antennas
    gamma1 = to_linear(ar1, factor=20)
    gamma2 = to_linear(ar2, factor=20)

    numerator = 4 * gamma1 * gamma2 - (1 - gamma1**2) * (1 - gamma2**2)
    denominator = (1 + gamma1**2) * (1 + gamma2**2)

    # Calculate polarization loss factor
    plf = 0.5 + 0.5 * (numerator / denominator)
    return safe_negate(to_dB(plf))


@enforce_units
def dish_gain(
    diameter: Length, frequency: Frequency, efficiency: Dimensionless
) -> Decibels:
    r"""
    Calculate the gain in dB of a parabolic dish antenna.

    Parameters
    ----------
    diameter : Length
        Dish diameter
    frequency : Frequency
        Frequency
    efficiency : Dimensionless
        Antenna efficiency (dimensionless)

    Returns
    -------
    Decibels
        Gain in decibels (dB)

    Raises
    ------
    ValueError
        If frequency is not positive
    """
    if frequency <= 0 * u.Hz:
        raise ValueError("Frequency must be positive")

    wl = wavelength(frequency)
    gain_linear = efficiency * (np.pi * diameter.to(u.m) / wl) ** 2
    return to_dB(gain_linear)
