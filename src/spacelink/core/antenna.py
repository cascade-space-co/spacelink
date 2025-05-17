"""
Core antenna calculation functions.
"""

import astropy.units as u  # Added import
import numpy as np  # Added import

# Assuming units.py is now in the same core directory
from .units import (
    Decibels,
    Dimensionless,
    Frequency,
    Length,
    wavelength,  # Need wavelength
    enforce_units,
    to_dB,
    to_linear,
    safe_negate,  # Add safe_negate import
)


@enforce_units
def polarization_loss(ar1: Decibels, ar2: Decibels) -> Decibels:
    r"""
    -VALIDATED-
    Calculate the polarization loss in dB between two antennas with given axial ratios.

    The polarization loss is calculated using the standard formula for polarization
    mismatch between two antennas with different axial ratios:

    .. math::
        PLF = \frac{1}{2} + \frac{1}{2} \frac{4 \gamma_1 \gamma_2 - (1-\gamma_1^2)(1-\gamma_2^2)}
             {(1+\gamma_1^2)(1+\gamma_2^2)}

    where:

    * :math:`\gamma_1` and :math:`\gamma_2` are the voltage axial ratios (linear, not dB)
    * PLF is the polarization loss factor (linear)

    The polarization loss in dB is then:

    .. math::
        L_{pol} = -10 \log_{10}(PLF)

    For circular polarization, the axial ratio is 0 dB, and for linear polarization,
    it is >40 dB.

    Parameters
    ----------
    ar1 : Quantity
        First antenna axial ratio in dB (amplitude ratio)
    ar2 : Quantity
        Second antenna axial ratio in dB (amplitude ratio)

    Returns
    -------
    Quantity
        Polarization loss in dB (positive value)

    Examples
    --------
    >>> polarization_loss(0 * u.dB, 0 * u.dB)  # Both circular
    <Quantity(0., 'dB')>
    >>> polarization_loss(0 * u.dB, 60 * u.dB)  # Circular to linear
    <Quantity(3.01, 'dB')>
    >>> polarization_loss(3 * u.dB, 3 * u.dB)  # Same elliptical polarization
    <Quantity(0.51, 'dB')>
    """
    # Polarization mismatch angle is omitted (assumed to be 90 degrees)
    # https://www.microwaves101.com/encyclopedias/polarization-mismatch-between-antennas
    gamma1 = to_linear(ar1, factor=20)
    gamma2 = to_linear(ar2, factor=20)

    numerator = 4 * gamma1 * gamma2 - (1 - gamma1**2) * (1 - gamma2**2)
    denominator = (1 + gamma1**2) * (1 + gamma2**2)

    # Calculate polarization loss factor
    plf = 0.5 + 0.5 * (numerator / denominator)

    # Convert to decibels and make positive (loss)
    return safe_negate(to_dB(plf))


@enforce_units
def dish_gain(
    diameter: Length, frequency: Frequency, efficiency: Dimensionless
) -> Decibels:
    r"""
    -VALIDATED-
    Calculate the gain in dB of a parabolic dish antenna.

    The gain of a parabolic dish antenna is given by:

    .. math::
        G = \eta \left(\frac{\pi D}{\lambda}\right)^2

    where:

    * :math:`\eta` is the efficiency factor (typically 0.55 to 0.70)
    * :math:`D` is the diameter of the dish
    * :math:`\lambda` is the wavelength

    In decibels:

    .. math::
        G_{dB} = 10\log_{10}(\eta) + 20\log_{10}(D) + 20\log_{10}(f)
        + 20\log_{10}\left(\frac{\pi}{c}\right)

    Parameters
    ----------
    diameter : Quantity
        Dish diameter
    frequency : Quantity
        Frequency
    efficiency : Quantity
        Antenna efficiency (dimensionless)

    Returns
    -------
    Quantity
        Gain in decibels (dB)

    Raises
    ------
    ValueError
        If frequency is not positive
    """
    # Added check back
    if frequency <= 0 * u.Hz:
        raise ValueError("Frequency must be positive")

    wl = wavelength(frequency)
    gain_linear = efficiency * (np.pi * diameter / wl) ** 2
    return to_dB(gain_linear)
