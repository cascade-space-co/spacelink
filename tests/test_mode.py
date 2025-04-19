"""Tests for the mode module."""

import pytest
from pint import Quantity

from pyradio.mode import Mode
from pyradio.units import Hz, MHz, db, dimensionless


def test_mode_initialization():
    """Test basic Mode initialization with valid parameters."""
    # Create a test mode
    # Test positive implementation loss as a positive dB value
    mode = Mode(
        name="BPSK 1/2",
        coding_scheme="Convolutional",
        modulation="BPSK",
        bits_per_symbol=Quantity(1, dimensionless),
        symbol_rate=Quantity(1, MHz),
        code_rate=0.5,
        spectral_efficiency=0.5,
        required_ebno=4.0,
        implementation_loss=2.0,
    )

    # Verify attributes
    assert mode.name == "BPSK 1/2"
    assert mode.coding_scheme == "Convolutional"
    assert mode.modulation == "BPSK"
    assert mode.bits_per_symbol.magnitude == pytest.approx(1.0)
    assert mode.symbol_rate == Quantity(1, MHz)
    assert mode.spectral_efficiency == pytest.approx(0.5)
    assert mode.required_ebno == pytest.approx(4.0)
    assert mode.implementation_loss == pytest.approx(2.0)
    assert mode.code_rate == pytest.approx(0.5)


def test_invalid_parameters():
    """Test that Mode initialization raises errors with invalid parameters."""
    # Empty name
    with pytest.raises(ValueError, match="Name must not be empty"):
        Mode(
            name="",
            coding_scheme="LDPC",
            modulation="QPSK",
            bits_per_symbol=Quantity(2, dimensionless),
            symbol_rate=Quantity(1, MHz),
            code_rate=0.5,
            spectral_efficiency=1.0,
            required_ebno=3.0,
        )

    # Empty coding scheme
    with pytest.raises(ValueError, match="Coding scheme must not be empty"):
        Mode(
            name="Test",
            coding_scheme="",
            modulation="QPSK",
            bits_per_symbol=Quantity(2, dimensionless),
            symbol_rate=Quantity(1, MHz),
            code_rate=0.5,
            spectral_efficiency=1.0,
            required_ebno=3.0,
        )

    # Empty modulation
    with pytest.raises(ValueError, match="Modulation must not be empty"):
        Mode(
            name="Test",
            coding_scheme="LDPC",
            modulation="",
            bits_per_symbol=Quantity(2, dimensionless),
            symbol_rate=Quantity(1, MHz),
            code_rate=0.5,
            spectral_efficiency=1.0,
            required_ebno=3.0,
        )

    # Non-positive spectral efficiency
    with pytest.raises(ValueError, match="Spectral efficiency must be positive"):
        Mode(
            name="Test",
            coding_scheme="LDPC",
            modulation="QPSK",
            bits_per_symbol=Quantity(2, dimensionless),
            symbol_rate=Quantity(1, MHz),
            code_rate=0.5,
            spectral_efficiency=0.0,
            required_ebno=3.0,
        )

    # Invalid code rate (too low)
    with pytest.raises(ValueError, match="Code rate must be between 0 and 1"):
        Mode(
            name="Test",
            coding_scheme="LDPC",
            modulation="QPSK",
            bits_per_symbol=Quantity(2, dimensionless),
            symbol_rate=Quantity(1, MHz),
            code_rate=0.0,
            spectral_efficiency=1.0,
            required_ebno=3.0,
        )

    # Invalid code rate (too high)
    with pytest.raises(ValueError, match="Code rate must be between 0 and 1"):
        Mode(
            name="Test",
            coding_scheme="LDPC",
            modulation="QPSK",
            bits_per_symbol=Quantity(2, dimensionless),
            symbol_rate=Quantity(1, MHz),
            code_rate=1.1,
            spectral_efficiency=1.0,
            required_ebno=3.0,
        )

    # Invalid implementation loss (should not be negative)
    with pytest.raises(ValueError, match="Implementation loss must be non-negative"):
        Mode(
            name="Test",
            coding_scheme="LDPC",
            modulation="QPSK",
            bits_per_symbol=Quantity(2, dimensionless),
            symbol_rate=Quantity(1, MHz),
            code_rate=0.5,
            spectral_efficiency=1.0,
            required_ebno=3.0,
            implementation_loss=-1.0,  # Negative loss is invalid
        )


def test_data_rate_calculation():
    """Test data rate calculation."""
    # Create a test mode
    mode = Mode(
        name="BPSK 1/2",
        coding_scheme="Convolutional",
        modulation="BPSK",
        bits_per_symbol=Quantity(1, dimensionless),
        symbol_rate=Quantity(1, MHz),
        code_rate=0.5,
        spectral_efficiency=0.5,
        required_ebno=4.0,
        implementation_loss=2.0,
    )

    # Calculate expected data rate: symbol_rate * bits_per_symbol * code_rate
    # 1 MHz * 1 * 0.5 = 0.5 MHz
    data_rate = mode.data_rate
    expected_value = 0.5 * 1e6  # 500 kHz

    # Convert to Hz for comparison
    assert data_rate.to(Hz).magnitude == pytest.approx(expected_value)


def test_bandwidth_calculation():
    """Test bandwidth calculation."""
    # Create a test mode
    mode = Mode(
        name="QPSK 3/4",
        coding_scheme="LDPC",
        modulation="QPSK",
        bits_per_symbol=Quantity(2, dimensionless),
        symbol_rate=Quantity(1, MHz),
        code_rate=0.75,
        spectral_efficiency=1.5,
        required_ebno=3.0,
        implementation_loss=1.0,
    )

    # Calculate expected bandwidth: symbol_rate / spectral_efficiency
    # 1 MHz / 1.5 = 0.667 MHz
    bandwidth = mode.bandwidth
    expected_bandwidth = Quantity(1.0 / 1.5, MHz)

    assert bandwidth.to(Hz).magnitude == pytest.approx(expected_bandwidth.to(Hz).magnitude)


def test_ebno_calculation():
    """Test Eb/N0 calculation from C/N."""
    # Create a test mode
    mode = Mode(
        name="QPSK 1/2",
        coding_scheme="Convolutional",
        modulation="QPSK",
        bits_per_symbol=Quantity(2, dimensionless),
        symbol_rate=Quantity(1, MHz),
        code_rate=0.5,
        spectral_efficiency=1.0,
        required_ebno=4.0,
        implementation_loss=2.0,
    )

    # With C/N of 10 dB and 2 bits per symbol, Eb/N0 should be 7 dB
    # Eb/N0 = C/N - 10*log10(bits_per_symbol)
    c_over_n = 10.0
    expected_ebno = c_over_n - db(2)  # 10 - 3.01 = 6.99 dB

    assert mode.ebno(c_over_n) == pytest.approx(expected_ebno, abs=0.01)


def test_margin_calculation():
    """Test margin calculation with directly calculated values."""
    # Create a test mode
    mode = Mode(
        name="BPSK 1/2",
        coding_scheme="Convolutional",
        modulation="BPSK",
        bits_per_symbol=Quantity(1, dimensionless),
        symbol_rate=Quantity(1, MHz),
        code_rate=0.5,
        spectral_efficiency=0.5,
        required_ebno=4.0,
        implementation_loss=2.0,
    )

    # Calculate manually what the implementation should do

    # Step 1: Use a known C/N ratio
    c_over_n = 8.0  # dB

    # Step 2: Calculate Eb/N0 manually
    # Eb/N0 = C/N - 10*log10(bits_per_symbol)
    # With bits_per_symbol = 1, log10(1) = 0, so Eb/N0 = C/N = 8.0 dB
    calculated_ebno = mode.ebno(c_over_n)
    expected_ebno = c_over_n - db(mode.bits_per_symbol.magnitude)

    # Verify that ebno calculation is correct
    assert calculated_ebno == pytest.approx(expected_ebno, abs=0.01)

    # Step 3: Calculate the margin
    # Margin = Eb/N0 - required_ebno - implementation_loss
    # With Eb/N0 = 8.0, required_ebno = 4.0, implementation_loss = 2.0
    # Expected margin = 8.0 - 4.0 - 2.0 = 2.0 dB
    expected_margin = calculated_ebno - mode.required_ebno - mode.implementation_loss

    # Call the margin method with C/N
    calculated_margin = mode.margin(c_over_n)

    # Verify that margin calculation is correct
    assert calculated_margin == pytest.approx(expected_margin, abs=0.01)


def test_str_representation():
    """Test string representation of Mode objects."""
    # Mode with all parameters
    mode = Mode(
        name="BPSK 1/2",
        coding_scheme="Convolutional",
        modulation="BPSK",
        bits_per_symbol=Quantity(1, dimensionless),
        symbol_rate=Quantity(1, MHz),
        code_rate=0.5,
        spectral_efficiency=0.5,
        required_ebno=4.0,
        implementation_loss=2.0,
    )

    str_repr = str(mode)
    assert "BPSK 1/2" in str_repr
    assert "BPSK with Convolutional" in str_repr
    assert "Spectral Efficiency: 0.500" in str_repr
    assert "Code Rate: 0.500" in str_repr
    assert "Required Eb/N0: 4.00 dB" in str_repr
    assert "Implementation Loss: 2.00 dB" in str_repr
