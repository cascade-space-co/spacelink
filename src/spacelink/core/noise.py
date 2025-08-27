r"""
Noise Power
-----------

The thermal noise power for a given bandwidth and temperature is:

.. math::
   P_n = k T B

where :math:`k` is Boltzmann's constant, :math:`T` is the noise temperature in Kelvin,
and :math:`B` is the bandwidth in Hz.

Density, is just:

.. math::
   P_n = k T

Noise Figure / Factor Modeling
------------------------------
The noise figure is a measure of the degradation of the signal-to-noise ratio (SNR) due
to the noise added by the amplifier, given an input noise temperature of T_0. If the
input noise temperature is not equal to T_0, noise figure cannot be used to measure the
degradation in SNR.

The relationships between noise temperature, noise figure, and noise factor are:

.. math::
    F = 1 + \frac{T}{T_0}

.. math::
    T = (F - 1) T_0

.. math::
    \text{NF}_{\text{dB}} = 10 \log_{10}(F)

.. math::
    F = 10^{\text{NF}_{\text{dB}}/10}

where :math:`T` is the noise temperature, :math:`T_0` is the reference temperature (290
K), :math:`F` is the noise factor, and :math:`\text{NF}_{\text{dB}}` is the noise figure
in dB.

Noise temperature and SNR
-------------------------
It is important to remember what the noise temperature is. The noise temperature of an
amplifier is the input referred noise temperature, meaning we model the amplifier as a
noiseless amplifier with a noisy resistor at the input with temperature :math:`T`. The
output noise temperature is therefore :math:`G T` where :math:`G` is the gain of the
amplifier. To accurately model the SNR through the amplifier chain, best practice is to
model the noise power separately from the signal power and then compute the SNR at the
output.
"""

import astropy.units as u
from astropy.constants import k_B as BOLTZMANN

from .units import (
    DecibelHertz,
    Decibels,
    Dimensionless,
    Frequency,
    Power,
    PowerDensity,
    Temperature,
    enforce_units,
    to_dB,
    to_linear,
)

T0 = 290.0 * u.K


@enforce_units
def noise_power(bandwidth: Frequency, temperature: Temperature = T0) -> Power:
    r"""
    Calculate the thermal noise power for a given bandwidth and temperature.

    Parameters
    ----------
    bandwidth : Frequency
        Bandwidth in Hz
    temperature : Temperature, optional
        Noise temperature in Kelvin (default: 290 K)

    Returns
    -------
    Power
        Noise power in Watts
    """
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
    temperature : Temperature
        Noise temperature in Kelvin

    Returns
    -------
    PowerDensity
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
    temperature : Temperature
        Noise temperature in Kelvin

    Returns
    -------
    Dimensionless
        Noise factor (dimensionless, linear)
    """
    return (1.0 + (temperature.to(u.K) / T0)).to(u.dimensionless)


@enforce_units
def noise_factor_to_temperature(noise_factor: Dimensionless) -> Temperature:
    r"""
    Convert noise factor (linear) to noise temperature.

    Parameters
    ----------
    noise_factor : Dimensionless
        Noise factor (dimensionless, linear)

    Returns
    -------
    Temperature
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
    noise_figure : Decibels
        Noise figure in dB

    Returns
    -------
    Temperature
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
    temperature : Temperature
        Noise temperature in Kelvin

    Returns
    -------
    Decibels
        Noise figure in dB
    """
    factor = temperature_to_noise_factor(temperature)
    return to_dB(factor)


@enforce_units
def ebn0_to_cn0(ebn0: Decibels, bitrate: Frequency) -> DecibelHertz:
    r"""
    Convert :math:`E_b/N_0` to :math:`C/N_0`.

    Parameters
    ----------
    ebn0 : Decibels
        :math:`E_b/N_0`
    bitrate : Frequency
        Bitrate

    Returns
    -------
    DecibelHertz
        :math:`C/N_0`
    """
    return ebn0 + bitrate.to(u.dBHz)


@enforce_units
def cn0_to_ebn0(cn0: DecibelHertz, bitrate: Frequency) -> Decibels:
    r"""
    Convert :math:`C/N_0` to :math:`E_b/N_0`.

    Parameters
    ----------
    cn0 : DecibelHertz
        :math:`C/N_0`
    bitrate : Frequency
        Bitrate

    Returns
    -------
    Decibels
        :math:`E_b/N_0`
    """
    return cn0 - bitrate.to(u.dBHz)
