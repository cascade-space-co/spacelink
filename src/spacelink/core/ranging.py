"""
Calculations related to two-way pseudo-noise (PN) radiometric ranging.

This module provides functions for calculating range ambiguity and power allocations
between residual carrier and modulated components.

References
----------
[1] 810-005 214, Rev. C "Pseudo-Noise and Regenerative Ranging"
    (part of the Deep Space Network Telecommunications Link Design Handbook)
    https://deepspace.jpl.nasa.gov/dsndocs/810-005/214/214C.pdf

[2] CCSDS 414.1-B-3 "Pseudo-Noise (PN) Ranging Systems Recommended Standard"
    https://ccsds.org/wp-content/uploads/gravity_forms/5-448e85c647331d9cbaf66c096458bdd5/2025/01//414x1b3e1.pdf

[3] CCSDS 414.0-G-2 "Pseudo-Noise (PN) Ranging Systems Informational Report"
    https://ccsds.org/wp-content/uploads/gravity_forms/5-448e85c647331d9cbaf66c096458bdd5/2025/01//414x0g2.pdf
"""

import enum
import math
import astropy.units as u
from astropy.coordinates import Angle
import astropy.constants as const
import numpy as np
from scipy.special import j0, j1

from .units import (
    Decibels,
    DecibelHertz,
    Dimensionless,
    Frequency,
    Distance,
    enforce_units,
)


class CommandMod(enum.Enum):
    """The type of command modulation used alongside ranging on the uplink."""

    BIPOLAR = enum.auto()
    SINE_SUBCARRIER = enum.auto()


# The DSN and CCSDS PN ranging codes all have the same length.
# [1] Equation (9).
# [2] Sections 3.2.2 and 3.2.3.
CODE_LENGTH = 1_009_470


@enforce_units
def pn_sequence_range_ambiguity(ranging_clock_rate: Frequency) -> Distance:
    r"""
    Compute the range ambiguity of the standard PN ranging sequences.

    Parameters
    ----------
    ranging_clock_rate : Frequency
        Rate of the ranging clock :math:`f_{RC}`. This is half the chip rate.

    Returns
    -------
    Distance
        The range ambiguity distance.

    References
    ----------
    [1] Equation (11).
    [3] p. 2-2.
    """
    return (CODE_LENGTH * const.c / (4 * ranging_clock_rate)).decompose()


@enforce_units
def chip_snr(ranging_clock_rate: Frequency, prn0: DecibelHertz) -> Decibels:
    r"""
    Compute the chip SNR :math:`2E_C/N_0` in decibels.

    Parameters
    ----------
    ranging_clock_rate : Frequency
        Rate of the ranging clock :math:`f_{RC}`. This is half the chip rate.
    prn0 : DecibelHertz
        The ranging signal-to-noise spectral density ratio :math:`P_R/N_0`.

    Returns
    -------
    Decibels
        The chip SNR :math:`2E_C/N_0`.

    References
    ----------
    [3] p. 2-3.
    """
    return prn0 - ranging_clock_rate.to(u.dB(u.Hz))


@enforce_units
def _suppression_factor(mod_idx: Angle, modulation: CommandMod) -> Dimensionless:
    r"""
    Compute the suppression factor :math:`S_{cmd}(\phi_{cmd})`.

    This is used in the expressions for uplink carrier and ranging power fractions.

    Parameters
    ----------
    mod_idx : Angle
        The RMS phase deviation by command signal :math:`\phi_{cmd}`.
    modulation : CommandModulation
        The command modulation type.

    Returns
    -------
    Dimensionless
        The suppression factor :math:`S_{cmd}(\phi_{cmd})`.

    References
    ----------
    [1] Equation (24).
    """
    mod_idx_rad = mod_idx.to(u.rad).value
    if modulation == CommandMod.BIPOLAR:
        return np.cos(mod_idx_rad) ** 2
    elif modulation == CommandMod.SINE_SUBCARRIER:
        return j0(math.sqrt(2) * mod_idx_rad) ** 2
    else:
        raise ValueError(f"Invalid command modulation type: {modulation}")


@enforce_units
def _modulation_factor(mod_idx: Angle, modulation: CommandMod) -> Dimensionless:
    r"""
    Compute the modulation factor :math:`M_{cmd}(\phi_{cmd})`.

    This is used in the expression for uplink data power fraction.

    Parameters
    ----------
    mod_idx : Angle
        The RMS phase deviation by command signal :math:`\phi_{cmd}`.
    modulation : CommandModulation
        The command modulation type.

    Returns
    -------
    Dimensionless
        The modulation factor :math:`M_{cmd}(\phi_{cmd})`.

    References
    ----------
    [1] Equation (25).
    """
    mod_idx_rad = mod_idx.to(u.rad).value
    if modulation == CommandMod.BIPOLAR:
        return np.sin(mod_idx_rad) ** 2
    elif modulation == CommandMod.SINE_SUBCARRIER:
        return 2 * j1(math.sqrt(2) * mod_idx_rad) ** 2
    else:
        raise ValueError(f"Invalid command modulation type: {modulation}")


@enforce_units
def uplink_carrier_to_total_power(
    mod_idx_ranging: Angle,
    mod_idx_cmd: Angle,
    modulation: CommandMod,
) -> Dimensionless:
    r"""
    Uplink ratio of residual carrier power to total power :math:`P_{C}/P_{T}`.

    Parameters
    ----------
    mod_idx_ranging : Angle
        The RMS phase deviation by ranging signal :math:`\phi_{r}`.
    mod_idx_cmd : Angle
        The RMS phase deviation by command signal :math:`\phi_{cmd}`.
    modulation : CommandModulation
        The command modulation type.

    Returns
    -------
    Dimensionless
        The ratio of residual carrier power to total power :math:`P_{C}/P_{T}`.

    References
    ----------
    [1] Equation (20).
    """
    return j0(
        math.sqrt(2) * mod_idx_ranging.to(u.rad).value
    ) ** 2 * _suppression_factor(mod_idx_cmd, modulation)


@enforce_units
def uplink_ranging_to_total_power(
    mod_idx_ranging: Angle,
    mod_idx_cmd: Angle,
    modulation: CommandMod,
) -> Dimensionless:
    r"""
    Uplink ratio of usable ranging power to total power :math:`P_{R}/P_{T}`.

    Parameters
    ----------
    mod_idx_ranging : Angle
        The RMS phase deviation by ranging signal :math:`\phi_{r}`.
    mod_idx_cmd : Angle
        The RMS phase deviation by command signal :math:`\phi_{cmd}`.
    modulation : CommandModulation
        The command modulation type.

    Returns
    -------
    Dimensionless
        The ratio of usable ranging power to total power :math:`P_{R}/P_{T}`.

    References
    ----------
    [1] Equation (20).
    """
    return (
        2
        * j1(math.sqrt(2) * mod_idx_ranging.to(u.rad).value) ** 2
        * _suppression_factor(mod_idx_cmd, modulation)
    )


@enforce_units
def uplink_data_to_total_power(
    mod_idx_ranging: Angle,
    mod_idx_cmd: Angle,
    modulation: CommandMod,
) -> Dimensionless:
    r"""
    Uplink ratio of usable data power to total power :math:`P_{D}/P_{T}`.

    Parameters
    ----------
    mod_idx_ranging : Angle
        The RMS phase deviation by ranging signal :math:`\phi_{r}`.
    mod_idx_cmd : Angle
        The RMS phase deviation by command signal :math:`\phi_{cmd}`.
    modulation : CommandModulation
        The command modulation type.

    Returns
    -------
    Dimensionless
        The ratio of usable data power to total power :math:`P_{D}/P_{T}`.

    References
    ----------
    [1] Equation (21).
    """
    return j0(math.sqrt(2) * mod_idx_ranging.to(u.rad).value) ** 2 * _modulation_factor(
        mod_idx_cmd, modulation
    )
