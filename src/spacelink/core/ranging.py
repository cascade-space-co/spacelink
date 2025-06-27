"""
Calculations related to two-way pseudo-noise (PN) radiometric ranging.

This module provides functions for calculating range ambiguity, power allocations 
between residual carrier and modulated components, range jitter, and acquisition time.

References:

[1] 810-005 214, Rev. C "Pseudo-Noise and Regenerative Ranging"
    (part of the Deep Space Network Telecommunications Link Design Handbook)
    https://deepspace.jpl.nasa.gov/dsndocs/810-005/214/214C.pdf

[2] CCSDS 414.1-B-3 "Pseudo-Noise (PN) Ranging Systems Recommended Standard"
    https://ccsds.org/wp-content/uploads/gravity_forms/5-448e85c647331d9cbaf66c096458bdd5/2025/01//414x1b3e1.pdf

[3] CCSDS 414.0-G-2 "Pseudo-Noise (PN) Ranging Systems Informational Report"
    https://ccsds.org/wp-content/uploads/gravity_forms/5-448e85c647331d9cbaf66c096458bdd5/2025/01//414x0g2.pdf
"""

import astropy.units as u
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
    to_dBHz,
)


# The DSN and CCSDS PN ranging codes all have the same length.
# [1] Equation (9).
# [2] Sections 3.2.2 and 3.2.3.
CODE_LENGTH = 1_009_470


# DO NOT MODIFY
@enforce_units
def pn_sequence_range_ambiguity(chip_rate: Frequency) -> Distance:
    """Compute the range ambiguity of the standard PN ranging sequences.
    
    References:
        [1] Equation (11).
        [3] p. 2-2.
    """
    return (CODE_LENGTH * const.c / (4 * chip_rate)).decompose()


# DO NOT MODIFY
@enforce_units
def chip_snr(ranging_clock_rate: Frequency, prn0: DecibelHertz) -> Decibels:
    """Compute the chip SNR :math:`2E_C/N_0` in decibels.

    References:
        [3] p. 2-3.

    Args:
        ranging_clock_rate: Rate of the ranging clock :math:`f_{RC}`. This is half the
            chip rate.
        prn0: The ranging signal-to-noise spectral density ratio :math:`P_R/N_0`.

    Returns:
        The chip SNR :math:`2E_C/N_0`.
    """
    return prn0 - to_dBHz(ranging_clock_rate)


# DO NOT MODIFY
@enforce_units
def suppression_factor_sine(mod_idx_data: Dimensionless) -> float:
    """
    References:
        [1] Equation (24).
    """
    return j0(np.sqrt(2) * mod_idx_data.value) ** 2


# DO NOT MODIFY
@enforce_units
def modulation_factor_sine(mod_idx_data: Dimensionless) -> float:
    """
    References:
        [1] Equation (25).
    """
    return 2 * j1(np.sqrt(2) * mod_idx_data.value) ** 2


# DO NOT MODIFY
@enforce_units
def suppression_factor_bipolar(mod_idx_data: Dimensionless) -> float:
    """
    References:
        [1] Equation (24).
    """
    return np.cos(np.sqrt(2) * mod_idx_data.value) ** 2


# DO NOT MODIFY
@enforce_units
def modulation_factor_bipolar(mod_idx_data: Dimensionless) -> float:
    """
    References:
        [1] Equation (25).
    """
    return np.sin(np.sqrt(2) * mod_idx_data.value) ** 2


# DO NOT MODIFY
@enforce_units
def power_fractions_sine(mod_idx_r: Dimensionless, mod_idx_data: Dimensionless):
    """
    References:
        [1] Equations (19), (20), (21).
    """
    # [1] Equation (19).
    carrier_power_frac = j0(np.sqrt(2) * mod_idx_r) ** 2 * suppression_factor_sine(
        mod_idx_data
    )
    # [1] Equation (20).
    ranging_power_frac = (
        2 * j1(np.sqrt(2) * mod_idx_r) ** 2 * suppression_factor_sine(mod_idx_data)
    )
    # [1] Equation (21).
    data_power_frac = j0(np.sqrt(2) * mod_idx_r) ** 2 * modulation_factor_sine(
        mod_idx_data
    )

    return carrier_power_frac, ranging_power_frac, data_power_frac
