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


@enforce_units
def coding_gain(
    uncoded_ber: Dimensionless,
    coded_ber: Dimensionless,
    code_rate: Dimensionless,
    *,
    is_power_limited: bool = True
) -> Decibels:
    r"""
    Calculate the coding gain of a channel code.

    The coding gain represents the reduction in required Eb/N0 to achieve
    the same bit error rate (BER) when using a specific code compared to
    an uncoded system. This calculation accounts for the code rate penalty.

    For power-limited systems:

    .. math::
        G_{coding} = 10 \log_{10}\left(\frac{E_b/N_0 \text{ uncoded}}{E_b/N_0 \text{ coded}}\right)

    For bandwidth-limited systems, the code rate is factored in:

    .. math::
        G_{coding} = 10 \log_{10}\left(\frac{E_b/N_0 \text{ uncoded}}{E_b/N_0 \text{ coded}} \cdot R\right)

    where :math:`R` is the code rate.

    Parameters
    ----------
    uncoded_ber : Dimensionless
        Bit error rate for the uncoded system
    coded_ber : Dimensionless
        Bit error rate for the coded system
    code_rate : Dimensionless
        Code rate (ratio of information bits to total bits)
    is_power_limited : bool, optional
        If True, calculate coding gain for power-limited systems.
        If False, calculate for bandwidth-limited systems.

    Returns
    -------
    Decibels
        Coding gain in dB

    Notes
    -----
    The coding gain calculation depends on whether the system is power-limited
    or bandwidth-limited. In power-limited systems, the raw coding gain is used.
    In bandwidth-limited systems, the code rate is factored in as a penalty.
    """
    # Convert to linear values if they are quantities
    uncoded_ber_val = uncoded_ber.to_value(u.dimensionless_unscaled)
    coded_ber_val = coded_ber.to_value(u.dimensionless_unscaled)
    code_rate_val = code_rate.to_value(u.dimensionless_unscaled)

    # Validate inputs
    if uncoded_ber_val <= 0 or uncoded_ber_val > 1:
        raise ValueError("Uncoded BER must be between 0 and 1")
    if coded_ber_val <= 0 or coded_ber_val > 1:
        raise ValueError("Coded BER must be between 0 and 1")
    if code_rate_val <= 0 or code_rate_val > 1:
        raise ValueError("Code rate must be between 0 and 1")

    # Calculate coding gain
    if is_power_limited:
        # For power-limited systems, don't include code rate penalty
        gain = uncoded_ber_val / coded_ber_val
    else:
        # For bandwidth-limited systems, include code rate penalty
        gain = (uncoded_ber_val / coded_ber_val) * code_rate_val

    return to_dB(gain * u.dimensionless)


@enforce_units
def required_ebno_coded(
    required_ebno_uncoded: Decibels,
    coding_gain: Decibels,
    code_rate: Dimensionless,
    implementation_loss: Optional[Decibels] = None,
) -> Decibels:
    r"""
    Calculate the required Eb/N0 for a coded system.

    This function calculates the required Eb/N0 for a coded system based on
    the required Eb/N0 for an uncoded system, the coding gain, and the code rate.

    .. math::
        \left(\frac{E_b}{N_0}\right)_{coded} = \left(\frac{E_b}{N_0}\right)_{uncoded} - G_{coding} + 10\log_{10}\left(\frac{1}{R}\right) + L_{impl}

    where:

    * :math:`G_{coding}` is the coding gain in dB
    * :math:`R` is the code rate
    * :math:`L_{impl}` is the implementation loss in dB

    Parameters
    ----------
    required_ebno_uncoded : Decibels
        Required Eb/N0 for the uncoded system in dB
    coding_gain : Decibels
        Coding gain in dB
    code_rate : Dimensionless
        Code rate (ratio of information bits to total bits)
    implementation_loss : Decibels, optional
        Implementation loss in dB (default is 0 dB)

    Returns
    -------
    Decibels
        Required Eb/N0 for the coded system in dB
    """
    # Convert code_rate to dB (negative value since it's a penalty)
    if code_rate <= 0 * u.dimensionless or code_rate > 1 * u.dimensionless:
        raise ValueError("Code rate must be between 0 and 1")

    # Use to_dB function to convert code_rate to dB, then negate it for the penalty
    code_rate_penalty = -to_dB(code_rate)

    # Set implementation loss to 0 dB if not provided
    if implementation_loss is None:
        implementation_loss = 0 * u.dB

    # Calculate required Eb/N0 for coded system
    return required_ebno_uncoded - coding_gain + code_rate_penalty + implementation_loss


@enforce_units
def theoretical_ber_bpsk(ebno: Decibels) -> Dimensionless:
    r"""
    Calculate the theoretical bit error rate (BER) for uncoded BPSK.

    For BPSK modulation in an AWGN channel, the theoretical BER is:

    .. math::
        P_b = Q\left(\sqrt{2\frac{E_b}{N_0}}\right) = \frac{1}{2}\text{erfc}\left(\sqrt{\frac{E_b}{N_0}}\right)

    where :math:`Q` is the Q-function and :math:`\text{erfc}` is the complementary error function.

    Parameters
    ----------
    ebno : Decibels
        Energy per bit to noise power spectral density ratio in dB

    Returns
    -------
    Dimensionless
        Theoretical bit error rate (BER)
    """
    # Convert Eb/N0 from dB to linear
    ebno_linear = to_linear(ebno).to_value(u.dimensionless_unscaled)

    # Calculate BER using complementary error function
    ber = 0.5 * special.erfc(np.sqrt(ebno_linear))

    return ber * u.dimensionless


@enforce_units
def theoretical_ber_qpsk(ebno: Decibels) -> Dimensionless:
    r"""
    Calculate the theoretical bit error rate (BER) for uncoded QPSK.

    For QPSK modulation in an AWGN channel, the theoretical BER is the same as BPSK:

    .. math::
        P_b = Q\left(\sqrt{2\frac{E_b}{N_0}}\right) = \frac{1}{2}\text{erfc}\left(\sqrt{\frac{E_b}{N_0}}\right)

    Parameters
    ----------
    ebno : Decibels
        Energy per bit to noise power spectral density ratio in dB

    Returns
    -------
    Dimensionless
        Theoretical bit error rate (BER)
    """
    # QPSK has the same BER as BPSK for the same Eb/N0
    return theoretical_ber_bpsk(ebno)


@enforce_units
def theoretical_ber_mfsk(ebno: Decibels, m: int) -> Dimensionless:
    r"""
    Calculate the theoretical bit error rate (BER) for uncoded M-FSK.

    For non-coherent M-FSK modulation in an AWGN channel, the theoretical BER is:

    .. math::
        P_b \approx \frac{M/2}{M-1} \exp\left(-\frac{E_b}{N_0} \frac{\log_2 M}{2}\right)

    Parameters
    ----------
    ebno : Decibels
        Energy per bit to noise power spectral density ratio in dB
    m : int
        Modulation order (number of frequency choices, must be a power of 2)

    Returns
    -------
    Dimensionless
        Theoretical bit error rate (BER)
    """
    # Check if M is a power of 2
    if m < 2 or (m & (m - 1)) != 0:
        raise ValueError("M must be a power of 2 (2, 4, 8, 16, etc.)")

    # Convert Eb/N0 from dB to linear
    ebno_linear = to_linear(ebno).to_value(u.dimensionless_unscaled)

    # Calculate log2(M)
    log2m = np.log2(m)

    # Calculate BER for non-coherent M-FSK
    ber = (m / 2) / (m - 1) * np.exp(-ebno_linear * log2m / 2)

    return ber * u.dimensionless


@enforce_units
def hamming_distance(code_rate: Dimensionless) -> int:
    r"""
    Estimate the minimum Hamming distance for a linear block code.

    This is a simplified estimation based on the Singleton bound, which states:

    .. math::
        d_{min} \leq n - k + 1

    where :math:`n` is the codeword length, :math:`k` is the message length,
    and :math:`d_{min}` is the minimum Hamming distance.

    For a code rate :math:`R = k/n`, we can estimate :math:`d_{min}` as:

    .. math::
        d_{min} \approx \lceil n \cdot (1 - R) \rceil

    This is a rough approximation and actual values depend on the specific code.

    Parameters
    ----------
    code_rate : Dimensionless
        Code rate (ratio of information bits to total bits)

    Returns
    -------
    int
        Estimated minimum Hamming distance
    """
    # Convert code_rate to a float
    r = code_rate.to_value(u.dimensionless_unscaled)

    if r <= 0 or r > 1:
        raise ValueError("Code rate must be between 0 and 1")

    # Assume a reasonable codeword length for estimation
    n = 100

    # Estimate minimum Hamming distance using Singleton bound
    d_min = np.ceil(n * (1 - r))

    return int(d_min)


@enforce_units
def error_correction_capability(hamming_distance: int) -> int:
    r"""
    Calculate the error correction capability of a code.

    A code with minimum Hamming distance :math:`d_{min}` can correct up to
    :math:`t` errors, where:

    .. math::
        t = \lfloor \frac{d_{min} - 1}{2} \rfloor

    Parameters
    ----------
    hamming_distance : int
        Minimum Hamming distance of the code

    Returns
    -------
    int
        Number of correctable errors
    """
    if hamming_distance < 2:
        raise ValueError("Hamming distance must be at least 2")

    # Calculate error correction capability
    t = (hamming_distance - 1) // 2

    return t


# Required Eb/N0 values (in dB) for concatenated coding schemes using PSK modulation
# The list index corresponds to the ErrorRate Enum value (0: 1e-3, 1: 1e-4, 2: 1e-5, 3: 1e-6)
_PSK_EBN0_VS_BER = {
    'CC(7,1/2) RS(255,223) I=5': [2.1, 2.25, 2.3, 2.4] * u.dB,
    'CC(7,2/3) RS(255,223) I=5': [2.8, 2.9, 3.05, 3.15] * u.dB,
    'CC(7,3/4) RS(255,223) I=5': [3.35, 3.5, 3.6, 3.65] * u.dB,
    'CC(7,5/6) RS(255,223) I=5': [4.1, 4.2, 4.3, 4.45] * u.dB,
    'CC(7,7/8) RS(255,223) I=5': [4.6, 4.7, 4.9, 5.0] * u.dB,
}

# Remove _BER_TARGET_MAP as it's no longer needed

# Remove enforce_units since Enum is not a Quantity
def required_ebno_for_psk_ber(ber: ErrorRate, coding_scheme: str) -> Decibels:
    """
    Look up the required Eb/N0 for a given target BER using concatenated coding schemes
    assuming PSK modulation.

    Uses pre-calculated values, likely from simulations or datasheets,
    for common concatenated coding schemes. Assumes the input error rate target
    corresponds to a Bit Error Rate (BER).

    .. note::
        The Eb/N0 values in the lookup table are typically associated with
        phase modulation schemes (e.g., BPSK, QPSK).

    Parameters
    ----------
    ber : ErrorRate
        Target error rate (must be a member of the ErrorRate Enum).
    coding_scheme : str, optional
        Coding scheme to use. Must be one of the keys in the internal lookup table.
        Defaults to 'CC(7,1/2) RS(255,223) I=5'.

    Returns
    -------
    Decibels
        Required Eb/N0 for the target BER and coding scheme in dB.

    Raises
    ------
    TypeError
        If ber is not an ErrorRate Enum member.
    ValueError
        If the coding scheme is not recognized.
    """
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
    # Directly add dB and dB(Hz) using astropy's capabilities
    return ebno + bit_rate.to(u.dB(u.Hz))


# Rename this function to avoid conflict
def coding_gain_from_ebno(ebno_uncoded: Decibels, ebno_coded: Decibels) -> Decibels:
    return ebno_uncoded - ebno_coded


def delta_c_n0(coding_gain: Decibels, coding_rate: Dimensionless) -> Decibels:
    """
    C/N0 requirement reduction due to coding
    """
    return coding_gain - to_dB(coding_rate)

