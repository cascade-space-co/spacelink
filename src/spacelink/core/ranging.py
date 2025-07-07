"""
Calculations related to two-way sequential or pseudo-noise (PN) radiometric ranging.

This module provides functions for calculating range ambiguity and power allocations
between residual carrier and modulated components.

References
----------
`[1]`_ 810-005 203, Rev. D "Sequential Ranging"

`[2]`_ 810-005 214, Rev. C "Pseudo-Noise and Regenerative Ranging"

`[3]`_ CCSDS 414.1-B-3 "Pseudo-Noise (PN) Ranging Systems Recommended Standard"

`[4]`_ CCSDS 414.0-G-2 "Pseudo-Noise (PN) Ranging Systems Informational Report"

.. _[1]: https://deepspace.jpl.nasa.gov/dsndocs/810-005/203/203D.pdf
.. _[2]: https://deepspace.jpl.nasa.gov/dsndocs/810-005/214/214C.pdf
.. _[3]: https://ccsds.org/wp-content/uploads/gravity_forms/\
5-448e85c647331d9cbaf66c096458bdd5/2025/01//414x1b3e1.pdf
.. _[4]: https://ccsds.org/wp-content/uploads/gravity_forms/\
5-448e85c647331d9cbaf66c096458bdd5/2025/01//414x0g2.pdf
"""

import enum
import math
import astropy.units as u
import astropy.constants as const
import numpy as np
from scipy.special import j0, j1

from .units import (
    Angle,
    Decibels,
    DecibelHertz,
    Dimensionless,
    Frequency,
    Distance,
    enforce_units,
)


class DataModulation(enum.Enum):
    """The type of data modulation used alongside ranging."""

    BIPOLAR = enum.auto()
    SINE_SUBCARRIER = enum.auto()


# The DSN and CCSDS PN ranging codes all have the same length.
# [2] Equation (9).
# [3] Sections 3.2.2 and 3.2.3.
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
    `[2]`_ Equation (11).

    `[4]`_ p. 2-2.
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
    `[4]`_ p. 2-3.
    """
    return prn0 - ranging_clock_rate.to(u.dB(u.Hz))


@enforce_units
def _suppression_factor(mod_idx: Angle, modulation: DataModulation) -> Dimensionless:
    r"""
    Compute the suppression factor :math:`S_{cmd}(\phi_{cmd})`.

    This is used in the expressions for carrier and ranging power fractions.

    Parameters
    ----------
    mod_idx : Angle
        The RMS phase deviation by command signal :math:`\phi_{cmd}`.
    modulation : DataModulation
        The data modulation type.

    Returns
    -------
    Dimensionless
        The suppression factor :math:`S_{cmd}(\phi_{cmd})`.

    References
    ----------
    `[1]`_ Equation (15).

    `[2]`_ Equation (24).
    """
    mod_idx_rad = mod_idx.value
    if modulation == DataModulation.BIPOLAR:
        suppression_factor = np.cos(mod_idx_rad) ** 2
    elif modulation == DataModulation.SINE_SUBCARRIER:
        suppression_factor = j0(math.sqrt(2) * mod_idx_rad) ** 2
    else:
        raise ValueError(f"Invalid data modulation type: {modulation}")
    return suppression_factor * u.dimensionless_unscaled


@enforce_units
def _modulation_factor(mod_idx: Angle, modulation: DataModulation) -> Dimensionless:
    r"""
    Compute the modulation factor :math:`M_{cmd}(\phi_{cmd})`.

    This is used in the expression for data power fraction.

    Parameters
    ----------
    mod_idx : Angle
        The RMS phase deviation by command signal :math:`\phi_{cmd}`.
    modulation : DataModulation
        The data modulation type.

    Returns
    -------
    Dimensionless
        The modulation factor :math:`M_{cmd}(\phi_{cmd})`.

    References
    ----------
    `[1]`_ Equation (16).

    `[2]`_ Equation (25).
    """
    mod_idx_rad = mod_idx.value
    if modulation == DataModulation.BIPOLAR:
        mod_factor = np.sin(mod_idx_rad) ** 2
    elif modulation == DataModulation.SINE_SUBCARRIER:
        mod_factor = 2 * j1(math.sqrt(2) * mod_idx_rad) ** 2
    else:
        raise ValueError(f"Invalid data modulation type: {modulation}")
    return mod_factor * u.dimensionless_unscaled


@enforce_units
def carrier_to_total_power(
    mod_idx_ranging: Angle,
    mod_idx_data: Angle,
    modulation: DataModulation,
) -> Dimensionless:
    r"""
    Ratio of residual carrier power to total power :math:`P_{C}/P_{T}`.

    This applies under the following conditions:

    * The ranging clock (chip pulse shape in the case of PN ranging) is a sinewave.
    * Uplink or regenerative downlink.

    This does not apply to the downlink case when a transparent (non-regenerative)
    transponder is used.

    Parameters
    ----------
    mod_idx_ranging : Angle
        The RMS phase deviation by ranging signal; :math:`\phi_{r}` for uplink or
        :math:`\theta_{rs}` for downlink.
    mod_idx_data : Angle
        The RMS phase deviation by data signal; :math:`\phi_{cmd}` for uplink or
        :math:`\theta_{tlm}` for downlink.
    modulation : DataModulation
        The data modulation type.

    Returns
    -------
    Dimensionless
        The ratio of residual carrier power to total power :math:`P_{C}/P_{T}`.

    References
    ----------
    `[1]`_ Equation (10) for sequential ranging uplink.

    `[2]`_ Equation (19) for PN ranging uplink, (50) for regenerative downlink.
    """
    return (
        j0(math.sqrt(2) * mod_idx_ranging.value) ** 2
        * _suppression_factor(mod_idx_data, modulation)
    ) * u.dimensionless_unscaled


@enforce_units
def ranging_to_total_power(
    mod_idx_ranging: Angle,
    mod_idx_data: Angle,
    modulation: DataModulation,
) -> Dimensionless:
    r"""
    Ratio of usable ranging power to total power :math:`P_{R}/P_{T}`.

    This applies under the following conditions:

    * The ranging clock (chip pulse shape in the case of PN ranging) is a sinewave.
    * Uplink or regenerative downlink.

    This does not apply to the downlink case when a transparent (non-regenerative)
    transponder is used.

    Parameters
    ----------
    mod_idx_ranging : Angle
        The RMS phase deviation by ranging signal; :math:`\phi_{r}` for uplink or
        :math:`\theta_{rs}` for downlink.
    mod_idx_data : Angle
        The RMS phase deviation by data signal; :math:`\phi_{cmd}` for uplink or
        :math:`\theta_{tlm}` for downlink.
    modulation : DataModulation
        The data modulation type.

    Returns
    -------
    Dimensionless
        The ratio of usable ranging power to total power :math:`P_{R}/P_{T}`.

    References
    ----------
    `[1]`_ Equation (11) for sequential ranging uplink.

    `[2]`_ Equation (20) for PN ranging uplink, (51) for regenerative downlink.
    """
    return (
        2
        * j1(math.sqrt(2) * mod_idx_ranging.value) ** 2
        * _suppression_factor(mod_idx_data, modulation)
    ) * u.dimensionless_unscaled


@enforce_units
def data_to_total_power(
    mod_idx_ranging: Angle,
    mod_idx_data: Angle,
    modulation: DataModulation,
) -> Dimensionless:
    r"""
    Ratio of usable data power to total power :math:`P_{D}/P_{T}`.

    This applies under the following conditions:

    * The ranging clock (chip pulse shape in the case of PN ranging) is a sinewave.
    * Uplink or regenerative downlink.

    This does not apply to the downlink case when a transparent (non-regenerative)
    transponder is used.

    Parameters
    ----------
    mod_idx_ranging : Angle
        The RMS phase deviation by ranging signal; :math:`\phi_{r}` for uplink or
        :math:`\theta_{rs}` for downlink.
    mod_idx_data : Angle
        The RMS phase deviation by data signal; :math:`\phi_{cmd}` for uplink or
        :math:`\theta_{tlm}` for downlink.
    modulation : DataModulation
        The data modulation type.

    Returns
    -------
    Dimensionless
        The ratio of usable data power to total power :math:`P_{D}/P_{T}`.

    References
    ----------
    `[1]`_ Equation (12) for sequential ranging uplink.

    `[2]`_ Equation (21) for PN ranging uplink, (52) for regenerative downlink.
    """
    return (
        j0(math.sqrt(2) * mod_idx_ranging.value) ** 2
        * _modulation_factor(mod_idx_data, modulation)
    ) * u.dimensionless_unscaled
