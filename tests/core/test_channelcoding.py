"""Tests for the channelcoding module."""

import pytest
import astropy.units as u
import numpy as np
from astropy.tests.helper import assert_quantity_allclose

from src.spacelink.core.channelcoding import (
    coding_gain,
    required_ebno_coded,
    theoretical_ber_bpsk,
    theoretical_ber_qpsk,
    theoretical_ber_mfsk,
    hamming_distance,
    error_correction_capability,
    required_c_n0,
    delta_c_n0,
    required_ebno_for_psk_ber,
    ErrorRate,
    get_code_rate_from_scheme,
)


def test_coding_gain_power_limited():
    """Test coding gain calculation for power-limited systems."""
    # Test with typical values
    uncoded_ber = 1e-3 * u.dimensionless
    coded_ber = 1e-6 * u.dimensionless
    code_rate = 0.5 * u.dimensionless
    
    # For power-limited systems, code rate is not factored in
    # Expected gain = 10*log10(uncoded_ber/coded_ber) = 10*log10(1000) ≈ 30 dB
    expected_gain = 30.0 * u.dB
    
    gain = coding_gain(uncoded_ber, coded_ber, code_rate, is_power_limited=True)
    assert_quantity_allclose(gain, expected_gain, rtol=1e-2)


def test_coding_gain_bandwidth_limited():
    """Test coding gain calculation for bandwidth-limited systems."""
    # Test with typical values
    uncoded_ber = 1e-3 * u.dimensionless
    coded_ber = 1e-6 * u.dimensionless
    code_rate = 0.5 * u.dimensionless
    
    # For bandwidth-limited systems, code rate is factored in
    # Expected gain = 10*log10((uncoded_ber/coded_ber)*code_rate) = 10*log10(1000*0.5) ≈ 27 dB
    expected_gain = 27.0 * u.dB
    
    gain = coding_gain(uncoded_ber, coded_ber, code_rate, is_power_limited=False)
    assert_quantity_allclose(gain, expected_gain, rtol=1e-2)


def test_required_ebno_coded():
    """Test required Eb/N0 calculation for coded systems."""
    # Test with typical values
    required_ebno_uncoded = 10.0 * u.dB
    coding_gain = 5.0 * u.dB
    code_rate = 0.5 * u.dimensionless
    implementation_loss = 2.0 * u.dB
    
    # Expected Eb/N0 = uncoded - coding_gain + code_rate_penalty + implementation_loss
    # code_rate_penalty = -10*log10(0.5) = 3.01 dB
    # Expected = 10 - 5 + 3.01 + 2 = 10.01 dB
    expected_ebno = 10.01 * u.dB
    
    ebno = required_ebno_coded(
        required_ebno_uncoded, coding_gain, code_rate, implementation_loss
    )
    assert_quantity_allclose(ebno, expected_ebno, rtol=1e-2)


def test_theoretical_ber_bpsk():
    """Test theoretical BER calculation for BPSK."""
    # Test with Eb/N0 = 10 dB
    ebno = 10.0 * u.dB
    
    # Expected BER for BPSK with Eb/N0 = 10 dB is approximately 3.87e-6
    expected_ber = 3.87e-6 * u.dimensionless
    
    ber = theoretical_ber_bpsk(ebno)
    assert_quantity_allclose(ber, expected_ber, rtol=1e-1)


def test_theoretical_ber_qpsk():
    """Test theoretical BER calculation for QPSK."""
    # Test with Eb/N0 = 10 dB
    ebno = 10.0 * u.dB
    
    # Expected BER for QPSK with Eb/N0 = 10 dB is the same as BPSK
    expected_ber = theoretical_ber_bpsk(ebno)
    
    ber = theoretical_ber_qpsk(ebno)
    assert_quantity_allclose(ber, expected_ber, rtol=1e-10)


def test_theoretical_ber_mfsk():
    """Test theoretical BER calculation for M-FSK."""
    # Test with Eb/N0 = 10 dB and M = 4
    ebno = 10.0 * u.dB
    m = 4
    
    # Expected BER for 4-FSK with Eb/N0 = 10 dB
    # Using the formula: (M/2)/(M-1) * exp(-Eb/N0 * log2(M)/2)
    ebno_linear = 10.0
    expected_ber = (m/2)/(m-1) * np.exp(-ebno_linear * np.log2(m)/2) * u.dimensionless
    
    ber = theoretical_ber_mfsk(ebno, m)
    assert_quantity_allclose(ber, expected_ber, rtol=1e-10)


def test_hamming_distance():
    """Test Hamming distance estimation."""
    # Test with code rate = 0.5
    code_rate = 0.5 * u.dimensionless
    
    # For code rate = 0.5, expected Hamming distance is approximately 50
    # (using n=100 in the implementation)
    expected_distance = 50
    
    distance = hamming_distance(code_rate)
    assert distance == expected_distance


def test_error_correction_capability():
    """Test error correction capability calculation."""
    # Test with Hamming distance = 7
    distance = 7
    
    # For Hamming distance = 7, expected correction capability is (7-1)/2 = 3
    expected_capability = 3
    
    capability = error_correction_capability(distance)
    assert capability == expected_capability


def test_invalid_inputs():
    """Test that functions raise appropriate errors for invalid inputs."""
    # Test coding_gain with invalid BER
    with pytest.raises(ValueError, match="Uncoded BER must be between 0 and 1"):
        coding_gain(-0.1 * u.dimensionless, 0.1 * u.dimensionless, 0.5 * u.dimensionless)
    
    with pytest.raises(ValueError, match="Coded BER must be between 0 and 1"):
        coding_gain(0.1 * u.dimensionless, 1.1 * u.dimensionless, 0.5 * u.dimensionless)
    
    # Test required_ebno_coded with invalid code rate
    with pytest.raises(ValueError, match="Code rate must be between 0 and 1"):
        required_ebno_coded(10 * u.dB, 5 * u.dB, 1.1 * u.dimensionless)
    
    # Test theoretical_ber_mfsk with invalid M
    with pytest.raises(ValueError, match="M must be a power of 2"):
        theoretical_ber_mfsk(10 * u.dB, 3)
    
    # Test error_correction_capability with invalid Hamming distance
    with pytest.raises(ValueError, match="Hamming distance must be at least 2"):
        error_correction_capability(1)

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

# Tests for required_ebno_for_psk_ber
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

# Tests for get_code_rate_from_scheme
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
