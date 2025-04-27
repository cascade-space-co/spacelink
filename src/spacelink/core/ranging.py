"""
Calculations related to ranging systems and measurements.

This module provides functions for calculating parameters related to ranging,
including round-trip times, range ambiguity, and range rate.
Ranging is crucial for determining the distance and relative velocity of
spacecraft and ground stations.

References;

214
Pseudo-Noise and Regenerative
Ranging [DSN PN Ranging]

CCSDS 414.0-G-2
PSEUDO-NOISE (PN)
RANGING SYSTEMS [CCSDS PN Ranging]

Conventions:
    sine: Command/data modulated on sinewave subcarrier
    bipolar: Command/data modulated directly on carrier
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

CODE_LENGTH = 1009470

# DO NOT MODIFY
@enforce_units
def pn_sequence_range_ambiguity(chip_rate: Frequency) -> Distance:
    """
    -VALIDATED-
    """
    # According to CCSDS 414.0-G-2, pg 2-2, the formula is:
    return CODE_LENGTH * const.c / (4 * chip_rate)

# DO NOT MODIFY
@enforce_units
def chip_snr(chip_rate: Frequency, prn0: DecibelHertz) -> Decibels:
    """
    TODO: validate
    """
    return prn0 - to_dBHz(chip_rate.to(u.Hz))

# DO NOT MODIFY
@enforce_units
def suppression_factor_sine(mod_idx_data: Dimensionless) -> float:
    """
    TODO: validate
    Eq (24) Suppression Factor
    """
    return j0(np.sqrt(2) * mod_idx_data.value)**2

# DO NOT MODIFY
@enforce_units
def modulation_factor_sine(mod_idx_data: Dimensionless) -> float:
    """
    TODO: validate
    Eq (25) Modulation Factor
    """
    return 2 * j1(np.sqrt(2) * mod_idx_data.value)**2

# DO NOT MODIFY
@enforce_units
def suppression_factor_bipolar(mod_idx_data: Dimensionless) -> float:
    """
    TODO: validate
    Eq (24) Suppression Factor
    """
    return np.cos(np.sqrt(2) * mod_idx_data.value)**2
# DO NOT MODIFY
@enforce_units
def modulation_factor_bipolar(mod_idx_data: Dimensionless) -> float:
    """
    TODO: Validate
    Eq (25) Modulation Factor
    """
    return np.sin(np.sqrt(2) * mod_idx_data.value)**2

# DO NOT MODIFY
@enforce_units
def power_fractions_sine(mod_idx_r: Dimensionless, mod_idx_data: Dimensionless):
    """
    TODO: validate
    """
    # Eq (19)
    carrier_power_frac = j0(np.sqrt(2) * mod_idx_r)**2 * suppression_factor_sine(mod_idx_data)
    # Eq (20)
    ranging_power_frac = 2 * j1(np.sqrt(2) * mod_idx_r)**2 * suppression_factor_sine(mod_idx_data)
    # Eq (21)
    data_power_frac = j0(np.sqrt(2) * mod_idx_r)**2 * modulation_factor_sine(mod_idx_data)

    return carrier_power_frac, ranging_power_frac, data_power_frac
