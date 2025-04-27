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

# uplink: 𝜙𝑟 = rms phase deviation by ranging signal, rad rms
# uplink: 𝜙𝑐𝑚𝑑 = rms phase deviation by command signal, rad rms
# downlink: 𝜃𝑟𝑠 = rms phase deviation by ranging signal (strong signal), rad rms
# downlink: 𝜃𝑟 = rms phase deviation by ranging signal, rad rms
# downlink: 𝜃𝑐𝑚𝑑 = rms phase deviation by feedthrough command signal, rad rms
# downlink: 𝜃𝑛 = rms phase deviation by noise, rad rms
# downlink: 𝜃𝑡𝑙𝑚 = rms phase deviation by telemetry signal, rad rms
# --- Ranging Calculations will go here --- #


def compute_power_fractions(phi_r, phi_cmd):
    """
    Compute the power fractions in the residual carrier (Pc),
    PN-ranging sidebands (Pr), and data subcarrier (Pcmd)
    for given RMS modulation indices phi_r (radians) and phi_cmd (radians).
    """
    x_r = np.sqrt(2) * phi_r
    x_c = np.sqrt(2) * phi_cmd

    # Residual carrier fraction: J0^2(sqrt(2)φr) · J0^2(sqrt(2)φcmd)
    Pc = j0(x_r)**2 * j0(x_c)**2

    # Ranging sidebands fraction: 2·J1^2(sqrt(2)φr) · J0^2(sqrt(2)φcmd)
    Pr = 2 * (j1(x_r)**2) * j0(x_c)**2

    # Data-subcarrier fraction: J0^2(sqrt(2)φr) · 2·J1^2(sqrt(2)φcmd)
    Pcmd = j0(x_r)**2 * 2 * (j1(x_c)**2)

    return Pc, Pr, Pcmd

@enforce_units
def pn_sequence_range_ambiguity(chip_rate: Frequency) -> Distance:
    # According to CCSDS 414.0-G-2, pg 2-2, the formula is:
    return CODE_LENGTH * const.c / (4 * chip_rate)


def chip_snr(chip_rate: Frequency, prn0: DecibelHertz) -> Decibels:
    return prn0 - to_dBHz(chip_rate.to(u.Hz))

def suppression_factor_sine(mod_idx_data: Dimensionless) -> float:
    """
    Eq (24) Suppression Factor
    """
    return j0(np.sqrt(2) * mod_idx_data.value)**2

def modulation_factor_sine(mod_idx_data: Dimensionless) -> float:
    """
    Eq (25) Modulation Factor
    """
    return 2 * j1(np.sqrt(2) * mod_idx_data.value)**2

def suppression_factor_bipolar(mod_idx_data: Dimensionless) -> float:
    """
    Eq (24) Suppression Factor
    """
    return np.cos(np.sqrt(2) * mod_idx_data.value)**2

def modulation_factor_bipolar(mod_idx_data: Dimensionless) -> float:
    """
    Eq (25) Modulation Factor
    """
    return np.sin(np.sqrt(2) * mod_idx_data.value)**2

def power_fractions_sine(mod_idx_r: Dimensionless, mod_idx_data: Dimensionless):

    # Eq (19)
    carrier_power_frac = j0(np.sqrt(2) * mod_idx_r)**2 * suppression_factor_sine(mod_idx_data)
    # Eq (20)
    ranging_power_frac = 2 * j1(np.sqrt(2) * mod_idx_r)**2 * suppression_factor_sine(mod_idx_data)
    # Eq (21)
    data_power_frac = j0(np.sqrt(2) * mod_idx_r)**2 * modulation_factor_sine(mod_idx_data)

    return carrier_power_frac, ranging_power_frac, data_power_frac
