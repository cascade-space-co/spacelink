"""
Channel coding calculations for communication systems.

This module provides functions for calculating various parameters related to
channel coding, including code rates, coding gain, and error correction capabilities.
Channel coding is essential for reliable communication in space applications where
signal-to-noise ratios may be low and bit errors are common.
"""

import astropy.units as u
import numpy as np
from scipy import special
from typing import Optional
from enum import Enum
import re # Add regex import

from .units import (
    Decibels,
    Dimensionless,
    Frequency,
    enforce_units,
    to_dB,
    to_linear,
    DecibelHertz,
)


# Define Enum for target error rate values - values are indices
class ErrorRate(Enum):
    E_NEG_3 = 0 # Corresponds to 1e-3
    E_NEG_4 = 1 # Corresponds to 1e-4
    E_NEG_5 = 2 # Corresponds to 1e-5
    E_NEG_6 = 3 # Corresponds to 1e-6


# Required Eb/N0 values (in dB) for concatenated coding schemes using PSK modulation
# The list index corresponds to the ErrorRate Enum value (0: 1e-3, 1: 1e-4, 2: 1e-5, 3: 1e-6)
_PSK_EBN0_VS_BER = {
    'CC(7,1/2) RS(255,223) I=5': [2.1, 2.25, 2.3, 2.4] * u.dB,
    'CC(7,2/3) RS(255,223) I=5': [2.8, 2.9, 3.05, 3.15] * u.dB,
    'CC(7,3/4) RS(255,223) I=5': [3.35, 3.5, 3.6, 3.65] * u.dB,
    'CC(7,5/6) RS(255,223) I=5': [4.1, 4.2, 4.3, 4.45] * u.dB,
    'CC(7,7/8) RS(255,223) I=5': [4.6, 4.7, 4.9, 5.0] * u.dB,
}

@enforce_units
def required_ebno_for_psk_ber(ber: ErrorRate, coding_scheme: str) -> Decibels:
    # Validate Enum type
    if not isinstance(ber, ErrorRate):
        raise TypeError(f"Input 'ber' must be an ErrorRate Enum member, got {type(ber)}")

    # Validate coding scheme
    if coding_scheme not in _PSK_EBN0_VS_BER:
        raise ValueError(f"Unknown coding scheme: {coding_scheme}. "
                         f"Available schemes: {list(_PSK_EBN0_VS_BER.keys())}")

    return _PSK_EBN0_VS_BER[coding_scheme][ber.value]


def get_code_rate_from_scheme(coding_scheme: str) -> Dimensionless:
    """Calculate the overall code rate from a coding scheme string.

    Parses concatenated coding scheme strings like 'CC(7,1/2) RS(255,223) I=5'
    to determine the overall code rate.

    The overall rate is the product of the Convolutional Code (CC) rate
    and the Reed-Solomon (RS) rate (k/n).

    Parameters
    ----------
    coding_scheme : str
        The coding scheme string.

    Returns
    -------
    Dimensionless
        The overall dimensionless code rate.

    Raises
    ------
    ValueError
        If the string format is not recognized or rates cannot be extracted.
    """
    # Regex to find CC rate like "X/Y"
    cc_match = re.search(r"CC\(\d+,(\d+)/(\d+)\)", coding_scheme)
    # Regex to find RS parameters like "(N,K)" - allow non-digits, int() will catch errors
    rs_match = re.search(r"RS\((.+?),(.+?)\)", coding_scheme)

    if not cc_match or not rs_match:
        raise ValueError(f"Could not parse CC and RS rates from scheme: {coding_scheme}")

    try:
        cc_k = int(cc_match.group(1))
        cc_n = int(cc_match.group(2))
        if cc_n == 0:
            raise ValueError("Convolutional code denominator cannot be zero")
        cc_rate = cc_k / cc_n

        rs_n = int(rs_match.group(1))
        rs_k = int(rs_match.group(2))
        if rs_n == 0:
            raise ValueError("Reed-Solomon block size (n) cannot be zero")
        if rs_k > rs_n:
             raise ValueError(f"Reed-Solomon k ({rs_k}) cannot be greater than n ({rs_n})")
        rs_rate = rs_k / rs_n

    except (IndexError, ValueError) as e:
        raise ValueError(f"Error parsing numbers from scheme '{coding_scheme}': {e}") from e

    overall_rate = cc_rate * rs_rate

    return overall_rate * u.dimensionless


def psk_bandwidth(symbol_rate: Frequency, alpha: Dimensionless) -> Frequency:
    return symbol_rate * (1 + alpha)


def required_c_n0(ebno: Decibels, bit_rate: Frequency) -> DecibelHertz:
    """
    Calculate the required carrier-to-noise power spectral density ratio (C/N0)
    for a given Eb/N0 and bit rate. The result is in dB-Hz.

    The formula is:
    C/N0 [dB-Hz] = Eb/N0 [dB] + 10*log10(bit_rate [Hz])

    This formula also holds for the coded case
    """
    return ebno + bit_rate.to(u.dB(u.Hz))


# Rename this function to avoid conflict
def coding_gain_from_ebno(ebno_uncoded: Decibels, ebno_coded: Decibels) -> Decibels:
    return ebno_uncoded - ebno_coded


def delta_c_n0(coding_gain: Decibels, coding_rate: Dimensionless) -> Decibels:
    """
    C/N0 requirement reduction due to coding
    """
    return coding_gain - to_dB(coding_rate)

