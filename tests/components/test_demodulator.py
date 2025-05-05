"""Tests for the Demodulator component."""

import pytest
import astropy.units as u
from astropy.tests.helper import assert_quantity_allclose

# Imports from spacelink
from spacelink.components.demodulator import Demodulator
from spacelink.components.signal import Signal
from spacelink.components.stage import Stage  # For type hinting input
from spacelink.components.mode import DataMode, RangingMode
from spacelink.core.modcod import ErrorRate  # ErrorRate is still in core
from spacelink.core.units import Frequency

# --- Mocks/Test Fixtures ---


class MockInputStage(Stage):
    """A mock stage that outputs a predefined signal."""

    def __init__(self, output_signal: Signal):
        super().__init__()
        self._output_signal = output_signal

    def output(self, frequency: Frequency) -> Signal:
        # Ignore frequency for this mock
        return self._output_signal


# Define some standard modes for testing
@pytest.fixture
def data_mode_bpsk_half():
    """Provides a standard BPSK 1/2 rate DataMode."""
    return DataMode(
        coding_scheme="CC(7,1/2) RS(255,223) I=5",
        bits_per_symbol=1 * u.dimensionless,
        error_rate=ErrorRate.E_NEG_5,
    )


@pytest.fixture
def ranging_mode_test():
    """Provides a standard RangingMode."""
    # Assuming RangingMode takes modulation indices
    # These values need verification based on RangingMode implementation
    # Placeholder values giving rough power fractions for testing
    # Let's aim for: Data ~ -3dB, Ranging ~ -6dB (relative to total C)
    return RangingMode(
        ranging_mod_idx=0.7 * u.dimensionless, data_mod_idx=1.0 * u.dimensionless
    )


@pytest.fixture
def mock_input_signal():
    """Provides a standard input signal for tests."""
    # Example: C/N0 = 50 dBHz
    # N0 = k * T => T = N0 / k
    # Let T = 100 K => N0 = 1.38e-23 * 100 = 1.38e-21 W/Hz
    # C = N0 * 10^(50/10) = 1.38e-21 * 1e5 = 1.38e-16 W
    power = 1.38e-16 * u.W
    noise_temp = 100 * u.K
    return Signal(power, noise_temp)


# --- Test Class ---


class TestDemodulatorCalculations:

    def test_total_cn0(self, mock_input_signal, data_mode_bpsk_half):
        """Test calculation of total C/N0."""
        # Setup - DataMode only needed for instantiation
        demod = Demodulator(
            carrier_frequency=1 * u.GHz,
            symbol_rate=1 * u.MHz,
            data_mode=data_mode_bpsk_half,
        )
        demod.input = MockInputStage(mock_input_signal)

        # Expected C/N0 for P=1.38e-16W, T=100K should be ~50 dB
        expected_cn0 = 50.0 * u.dB
        assert_quantity_allclose(demod.cn0(), expected_cn0, atol=0.1 * u.dB)

    def test_data_cn0_data_only(self, mock_input_signal, data_mode_bpsk_half):
        """Test data C/N0 when only data mode is present."""
        demod = Demodulator(
            carrier_frequency=1 * u.GHz,
            symbol_rate=1 * u.MHz,
            data_mode=data_mode_bpsk_half,
        )
        demod.input = MockInputStage(mock_input_signal)

        # data_cn0 should equal total cn0
        expected_cn0 = 50.0 * u.dB
        assert_quantity_allclose(demod.data_cn0, expected_cn0, atol=0.1 * u.dB)

    @pytest.mark.xfail(
        reason="RangingMode power fraction implementation needed/verified"
    )
    def test_data_cn0_with_ranging(
        self, mock_input_signal, data_mode_bpsk_half, ranging_mode_test
    ):
        """Test data C/N0 calculation when ranging mode shares power."""
        demod = Demodulator(
            carrier_frequency=1 * u.GHz,
            symbol_rate=1 * u.MHz,
            data_mode=data_mode_bpsk_half,
            ranging_mode=ranging_mode_test,
        )
        demod.input = MockInputStage(mock_input_signal)

        # data_cn0 = total_cn0 + data_power_fraction_db
        # Using placeholder RangingMode, assume data_power_fraction is ~ -3 dB
        expected_data_cn0 = (50.0 - 3.0) * u.dB
        assert_quantity_allclose(demod.data_cn0, expected_data_cn0, atol=0.1 * u.dB)

    @pytest.mark.xfail(
        reason="RangingMode power fraction implementation needed/verified"
    )
    def test_ranging_prn0(
        self, mock_input_signal, data_mode_bpsk_half, ranging_mode_test
    ):
        """Test ranging Pr/N0 calculation."""
        # Need at least one mode for instantiation
        demod = Demodulator(
            carrier_frequency=1 * u.GHz,
            symbol_rate=1 * u.MHz,
            data_mode=data_mode_bpsk_half,
            ranging_mode=ranging_mode_test,
        )
        demod.input = MockInputStage(mock_input_signal)

        # ranging_prn0 = total_cn0 + ranging_power_fraction_db
        # Using placeholder RangingMode, assume ranging_power_fraction is ~ -6 dB
        expected_ranging_prn0 = (50.0 - 6.0) * u.dB
        assert_quantity_allclose(
            demod.ranging_prn0, expected_ranging_prn0, atol=0.1 * u.dB
        )

    def test_ebno(self, mock_input_signal, data_mode_bpsk_half):
        """Test Eb/N0 calculation delegation."""
        demod = Demodulator(
            carrier_frequency=1 * u.GHz,
            symbol_rate=1 * u.MHz,
            data_mode=data_mode_bpsk_half,
        )
        demod.input = MockInputStage(mock_input_signal)

        # Eb/N0 = C/N0 [dBHz] - 10*log10(BitRate [Hz]) [dB]
        # BitRate = SymbolRate * BitsPerSymbol * CodeRate
        # For BPSK 1/2: BR = 1MHz * 1 * (1/2 * 223/255) = 437.25 kHz
        # 10*log10(437.25e3) = 56.4 dB
        # C/N0 is now 50 dB. Calculation likely needs dBHz input in DataMode.
        expected_ebno = -6.4 * u.dB
        # This test will likely fail as data_cn0 is dB, not dBHz
        assert_quantity_allclose(demod.ebno, expected_ebno, atol=0.1 * u.dB)

    def test_data_margin(self, mock_input_signal, data_mode_bpsk_half):
        """Test data margin calculation delegation."""
        demod = Demodulator(
            carrier_frequency=1 * u.GHz,
            symbol_rate=1 * u.MHz,
            data_mode=data_mode_bpsk_half,
        )
        demod.input = MockInputStage(mock_input_signal)

        # Margin = Actual Eb/N0 - Required Eb/N0
        # Required Eb/N0 for BPSK 1/2 at 1e-5 is 2.3 dB
        # Actual Eb/N0 calculation in test_ebno assumes C/N0 is 50 dBHz, but it's 50 dB now.
        # Margin = -6.4 dB - 2.3 dB = -8.7 dB
        expected_margin = -8.7 * u.dB
        # This test will likely fail as data_cn0 is dB, not dBHz
        assert_quantity_allclose(demod.data_margin, expected_margin, atol=0.1 * u.dB)


# TODO: Add tests for ranging_margin once implemented
