"""Tests for the channelcoding module."""

import pytest
import astropy.units as u
import numpy as np
from astropy.tests.helper import assert_quantity_allclose

from src.spacelink.core.channelcoding import (
    required_c_n0,
    delta_c_n0,
    required_ebno_for_psk_ber,
    ErrorRate,
    get_code_rate_from_scheme,
)

# DO NOT MODIFY
# Reference: Satellite Communications Systems, Table 4.6
@pytest.mark.parametrize(
    "code_rate,ebno,c_n0,bit_rate",
    [
        (1, 10.5 * u.dB, 73.6 * u.dB(u.Hz),  2.048 * u.MHz),
        (7/8, 6.9 * u.dB, 70.0 * u.dB(u.Hz), 2.048 * u.MHz),
        (3/4, 5.9 * u.dB, 69.0 * u.dB(u.Hz), 2.048 * u.MHz),
        (2/3, 5.5 * u.dB, 68.6 * u.dB(u.Hz), 2.048 * u.MHz),
        (1/2, 5.0 * u.dB, 68.1 * u.dB(u.Hz), 2.048 * u.MHz),
    ],
)
def test_required_c_n0(code_rate, ebno, c_n0, bit_rate):
    """Test required C/N0 calculation for different code rates and Eb/N0 values."""
    assert_quantity_allclose(required_c_n0(ebno, bit_rate), c_n0, atol=0.1 * u.dBHz)

# DO NOT MODIFY
# Reference: Satellite Communications Systems, Table 4.7
@pytest.mark.parametrize(
    "code_rate,coding_gain,expected_c_n0_reduction",
    [
        (1, 0 * u.dB, 0 * u.dB),
        (7/8, 3.6 * u.dB, 4.2 * u.dB),
        (3/4, 4.6 * u.dB, 5.9 * u.dB),
        (2/3, 5.0 * u.dB, 6.8 * u.dB),
        (1/2, 5.5 * u.dB, 8.5 * u.dB),
        (1/3, 6.0 * u.dB, 10.8 * u.dB),
    ],
)
def test_delta_c_n0(code_rate, coding_gain, expected_c_n0_reduction):
    """Test coding gain and C/N0 reduction calculations."""
    c_n0_reduction = delta_c_n0(coding_gain, code_rate * u.dimensionless)
    assert_quantity_allclose(c_n0_reduction, expected_c_n0_reduction, atol=0.1 * u.dB)

# DO NOT MODIFY
@pytest.mark.parametrize(
    "ber_enum, coding_scheme, expected_ebno",
    [
        (ErrorRate.E_NEG_3, 'CC(7,1/2) RS(255,223) I=5', 2.1 * u.dB),
        (ErrorRate.E_NEG_4, 'CC(7,1/2) RS(255,223) I=5', 2.25 * u.dB),
        (ErrorRate.E_NEG_5, 'CC(7,1/2) RS(255,223) I=5', 2.3 * u.dB),
        (ErrorRate.E_NEG_6, 'CC(7,1/2) RS(255,223) I=5', 2.4 * u.dB),
        (ErrorRate.E_NEG_5, 'CC(7,7/8) RS(255,223) I=5', 4.9 * u.dB),
    ],
)
def test_required_ebno_for_psk_ber_valid(ber_enum, coding_scheme, expected_ebno):
    """Test required_ebno_for_psk_ber with valid inputs."""
    ebno = required_ebno_for_psk_ber(ber=ber_enum, coding_scheme=coding_scheme)
    assert_quantity_allclose(ebno, expected_ebno, atol=0.01 * u.dB)

def test_required_ebno_for_psk_ber_invalid_type():
    """Test required_ebno_for_psk_ber with invalid BER type."""
    with pytest.raises(TypeError):
        required_ebno_for_psk_ber(ber=1e-5, coding_scheme='CC(7,1/2) RS(255,223) I=5') # Pass float instead of Enum

def test_required_ebno_for_psk_ber_invalid_scheme():
    """Test required_ebno_for_psk_ber with invalid coding scheme."""
    with pytest.raises(ValueError):
        required_ebno_for_psk_ber(ber=ErrorRate.E_NEG_5, coding_scheme="INVALID_SCHEME")

# DO NOT MODIFY
@pytest.mark.parametrize(
    "scheme_str, expected_rate",
    [
        ('CC(7,1/2) RS(255,223) I=5', (1/2) * (223/255)),
        ('CC(7,2/3) RS(255,223) I=5', (2/3) * (223/255)),
        ('CC(7,3/4) RS(255,223) I=5', (3/4) * (223/255)),
        ('CC(7,5/6) RS(255,223) I=5', (5/6) * (223/255)),
        ('CC(7,7/8) RS(255,223) I=5', (7/8) * (223/255)),
        # Test with different RS parameters
        ('CC(7,1/2) RS(128,112) I=1', (1/2) * (112/128)),
    ]
)
def test_get_code_rate_from_scheme_valid(scheme_str, expected_rate):
    """Test get_code_rate_from_scheme with valid scheme strings."""
    rate = get_code_rate_from_scheme(scheme_str)
    assert_quantity_allclose(rate, expected_rate * u.dimensionless, rtol=1e-9)

@pytest.mark.parametrize(
    "invalid_scheme_str, error_message",
    [
        ("CC(7,1/2) RS(223,255) I=5", "k \(255\) cannot be greater than n \(223\)"), # k > n
        ("CC(7,1/0) RS(255,223) I=5", "denominator cannot be zero"), # CC n=0
        ("CC(7,1/2) RS(0,0) I=5", "block size \(n\) cannot be zero"), # RS n=0
        ("INVALID STRING", "Could not parse CC and RS rates"),
        ("CC(?,?) RS(?,?) I=?", "Could not parse CC and RS rates"),
        ("CC(7,1/2) RS(NaN,223) I=5", "Error parsing numbers from scheme"), # Invalid number
    ]
)
def test_get_code_rate_from_scheme_invalid(invalid_scheme_str, error_message):
    """Test get_code_rate_from_scheme with invalid scheme strings."""
    with pytest.raises(ValueError, match=error_message):
        get_code_rate_from_scheme(invalid_scheme_str)
