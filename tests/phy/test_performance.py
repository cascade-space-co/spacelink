from fractions import Fraction

import astropy.units as u
import numpy as np
import pytest
from astropy.tests.helper import assert_quantity_allclose

from spacelink.phy.mode import Code, CodeChain, LinkMode, Modulation
from spacelink.phy.performance import (
    ErrorMetric,
    ModePerformanceCurve,
    ModePerformanceThreshold,
)

# Test constants
STANDARD_CURVE_POINTS = [(0.0, 1e-1), (1.0, 3e-2), (2.0, 5e-3), (3.0, 1e-4)]
CODED_CURVE_POINTS = [(0.0, 1e-0), (1.0, 1e-1), (2.0, 1e-2), (3.0, 1e-3)]
UNCODED_CURVE_POINTS = [(1.0, 1e-0), (3.0, 1e-1), (5.0, 1e-2), (7.0, 1e-3)]
DVB_S2_THRESHOLD_EBN0 = 5.0
DVB_S2_THRESHOLD_ERROR_RATE = 1e-7


# Fixtures
@pytest.fixture
def basic_bpsk_mode():
    """Basic BPSK uncoded mode for testing."""
    modulation = Modulation(name="BPSK", bits_per_symbol=1)
    coding = CodeChain(codes=[Code(name="uncoded", rate=Fraction(1))])
    return LinkMode(id="TEST", modulation=modulation, coding=coding)


@pytest.fixture
def qpsk_ldpc_mode():
    """QPSK mode with LDPC 1/4 coding (DVB-S2 style)."""
    modulation = Modulation(name="QPSK", bits_per_symbol=2)
    coding = CodeChain(codes=[Code(name="LDPC", rate=Fraction(1, 4))])
    return LinkMode(id="DVB_S2_QPSK_1_4", modulation=modulation, coding=coding)


@pytest.fixture
def qpsk_uncoded_mode():
    """QPSK uncoded mode for comparison tests."""
    modulation = Modulation(name="QPSK", bits_per_symbol=2)
    coding = CodeChain(codes=[Code(name="uncoded", rate=Fraction(1))])
    return LinkMode(id="UNCODED_QPSK", modulation=modulation, coding=coding)


class TestModePerformanceCurve:
    def test_creation_with_points(self, basic_bpsk_mode):
        model = ModePerformanceCurve(
            modes=[basic_bpsk_mode],
            metric=ErrorMetric.BER,
            points=STANDARD_CURVE_POINTS,
        )

        assert model.points == STANDARD_CURVE_POINTS

    def test_ebn0_to_error_rate_exact_points(self, basic_bpsk_mode):
        model = ModePerformanceCurve(
            modes=[basic_bpsk_mode],
            metric=ErrorMetric.BER,
            points=STANDARD_CURVE_POINTS,
        )

        assert_quantity_allclose(
            model.ebn0_to_error_rate(0.0 * u.dB(1)), 1e-1 * u.dimensionless
        )
        assert_quantity_allclose(
            model.ebn0_to_error_rate(1.0 * u.dB(1)), 3e-2 * u.dimensionless
        )
        assert_quantity_allclose(
            model.ebn0_to_error_rate(2.0 * u.dB(1)), 5e-3 * u.dimensionless
        )
        assert_quantity_allclose(
            model.ebn0_to_error_rate(3.0 * u.dB(1)), 1e-4 * u.dimensionless
        )

    def test_ebn0_to_error_rate_between_points(self, basic_bpsk_mode):
        model = ModePerformanceCurve(
            modes=[basic_bpsk_mode],
            metric=ErrorMetric.BER,
            points=STANDARD_CURVE_POINTS,
        )

        ber_1_5 = model.ebn0_to_error_rate(1.5 * u.dB(1)).value
        assert 5e-3 < ber_1_5 < 3e-2

    def test_error_rate_to_ebn0_exact_points(self, basic_bpsk_mode):
        model = ModePerformanceCurve(
            modes=[basic_bpsk_mode],
            metric=ErrorMetric.BER,
            points=STANDARD_CURVE_POINTS,
        )

        assert_quantity_allclose(
            model.error_rate_to_ebn0(1e-1 * u.dimensionless),
            0.0 * u.dB(1),
            atol=1e-15 * u.dB(1),
        )
        assert_quantity_allclose(
            model.error_rate_to_ebn0(3e-2 * u.dimensionless), 1.0 * u.dB(1)
        )
        assert_quantity_allclose(
            model.error_rate_to_ebn0(5e-3 * u.dimensionless), 2.0 * u.dB(1)
        )
        assert_quantity_allclose(
            model.error_rate_to_ebn0(1e-4 * u.dimensionless), 3.0 * u.dB(1)
        )

    def test_error_rate_to_ebn0_between_points(self, basic_bpsk_mode):
        model = ModePerformanceCurve(
            modes=[basic_bpsk_mode],
            metric=ErrorMetric.BER,
            points=STANDARD_CURVE_POINTS,
        )

        ber_1_5 = model.ebn0_to_error_rate(1.5 * u.dB(1)).value
        ebn0_1_5 = model.error_rate_to_ebn0(ber_1_5 * u.dimensionless).value
        assert 1.0 < ebn0_1_5 < 2.0

    def test_interpolation_with_array_inputs(self, basic_bpsk_mode):
        model = ModePerformanceCurve(
            modes=[basic_bpsk_mode],
            metric=ErrorMetric.BER,
            points=STANDARD_CURVE_POINTS,
        )

        ebn0_array = np.array([0.0, 1.0, 2.0, 3.0]) * u.dB(1)
        error_rates = model.ebn0_to_error_rate(ebn0_array)
        assert_quantity_allclose(
            error_rates, np.array([1e-1, 3e-2, 5e-3, 1e-4]) * u.dimensionless
        )

        error_rate_array = np.array([1e-1, 3e-2, 5e-3, 1e-4]) * u.dimensionless
        ebn0_values = model.error_rate_to_ebn0(error_rate_array)
        assert_quantity_allclose(
            ebn0_values,
            np.array([0.0, 1.0, 2.0, 3.0]) * u.dB(1),
            atol=1e-15 * u.dB(1),
        )

    def test_coding_gain(self, basic_bpsk_mode):
        coded_model = ModePerformanceCurve(
            modes=[basic_bpsk_mode],
            metric=ErrorMetric.BER,
            points=CODED_CURVE_POINTS,
        )

        uncoded_model = ModePerformanceCurve(
            modes=[basic_bpsk_mode],
            metric=ErrorMetric.BER,
            points=UNCODED_CURVE_POINTS,
        )

        gain_1e0 = coded_model.coding_gain(uncoded_model, 1e-0 * u.dimensionless)
        assert_quantity_allclose(gain_1e0, 1.0 * u.dB(1))

        gain_1e1 = coded_model.coding_gain(uncoded_model, 1e-1 * u.dimensionless)
        assert_quantity_allclose(gain_1e1, 2.0 * u.dB(1))

        gain_1e2 = coded_model.coding_gain(uncoded_model, 1e-2 * u.dimensionless)
        assert_quantity_allclose(gain_1e2, 3.0 * u.dB(1))

        error_rate_array = np.array([1e-0, 1e-1, 1e-2]) * u.dimensionless
        gains = coded_model.coding_gain(uncoded_model, error_rate_array)
        assert_quantity_allclose(gains, np.array([1.0, 2.0, 3.0]) * u.dB(1))

    def test_edge_cases(self, basic_bpsk_mode):
        """Test NaN return for out-of-bounds interpolation values."""
        model = ModePerformanceCurve(
            modes=[basic_bpsk_mode],
            metric=ErrorMetric.BER,
            points=STANDARD_CURVE_POINTS,
        )

        assert np.isnan(model.ebn0_to_error_rate(-1.0 * u.dB(1)).value)
        assert np.isnan(model.ebn0_to_error_rate(4.0 * u.dB(1)).value)
        assert np.isnan(model.error_rate_to_ebn0(1e-0 * u.dimensionless).value)
        assert np.isnan(model.error_rate_to_ebn0(1e-5 * u.dimensionless).value)

    def test_insufficient_points(self, basic_bpsk_mode):
        """Test that curves with fewer than 2 points are rejected."""
        with pytest.raises(ValueError):
            ModePerformanceCurve(
                modes=[basic_bpsk_mode],
                metric=ErrorMetric.BER,
                points=[],
            )

        with pytest.raises(ValueError):
            ModePerformanceCurve(
                modes=[basic_bpsk_mode],
                metric=ErrorMetric.BER,
                points=[(1.0, 1e-2)],
            )

    def test_multiple_modes(self, basic_bpsk_mode):
        modulation2 = Modulation(name="QPSK", bits_per_symbol=2)
        coding2 = CodeChain(codes=[Code(name="uncoded", rate=Fraction(1))])
        mode2 = LinkMode(id="TEST2", modulation=modulation2, coding=coding2)

        model = ModePerformanceCurve(
            modes=[basic_bpsk_mode, mode2],
            metric=ErrorMetric.BER,
            points=STANDARD_CURVE_POINTS,
        )

        assert len(model.modes) == 2
        assert model.modes[0].id == "TEST"
        assert model.modes[1].id == "TEST2"

    def test_coding_gain_metric_mismatch_error(self, basic_bpsk_mode):
        """Test that coding_gain raises ValueError when metrics don't match."""
        ber_model = ModePerformanceCurve(
            modes=[basic_bpsk_mode],
            metric=ErrorMetric.BER,
            points=STANDARD_CURVE_POINTS,
        )

        fer_model = ModePerformanceCurve(
            modes=[basic_bpsk_mode],
            metric=ErrorMetric.FER,
            points=STANDARD_CURVE_POINTS,
        )

        with pytest.raises(ValueError):
            ber_model.coding_gain(fer_model, 1e-2 * u.dimensionless)

    @pytest.mark.parametrize(
        "points,description",
        [
            ([(3.0, 1e-4), (2.0, 5e-3), (1.0, 3e-2), (0.0, 1e-1)], "decreasing"),
            ([(0.0, 1e-1), (2.0, 5e-3), (1.0, 3e-2), (3.0, 1e-4)], "mixed"),
            ([(0.0, 1e-1), (1.0, 3e-2), (1.0, 5e-3), (3.0, 1e-4)], "duplicate"),
        ],
    )
    def test_unsorted_points_rejected(self, basic_bpsk_mode, points, description):
        """Test that unsorted Eb/N0 points are rejected during construction."""
        with pytest.raises(ValueError):
            ModePerformanceCurve(
                modes=[basic_bpsk_mode], metric=ErrorMetric.BER, points=points
            )

    def test_sorted_points_accepted(self, basic_bpsk_mode):
        """Test that properly sorted points are accepted."""
        model = ModePerformanceCurve(
            modes=[basic_bpsk_mode],
            metric=ErrorMetric.BER,
            points=STANDARD_CURVE_POINTS,
        )
        assert model.points == STANDARD_CURVE_POINTS

    @pytest.mark.parametrize(
        "points,description",
        [
            ([(0.0, 1e-4), (1.0, 3e-3), (2.0, 5e-2), (3.0, 1e-1)], "increasing"),
            ([(0.0, 1e-2), (1.0, 1e-2), (2.0, 1e-2), (3.0, 1e-2)], "constant"),
            ([(0.0, 1e-1), (1.0, 3e-2), (2.0, 5e-2), (3.0, 1e-4)], "non-monotonic"),
        ],
    )
    def test_non_decreasing_error_values_rejected(
        self, basic_bpsk_mode, points, description
    ):
        """Test that non-decreasing error values are rejected during construction."""
        with pytest.raises(ValueError):
            ModePerformanceCurve(
                modes=[basic_bpsk_mode], metric=ErrorMetric.BER, points=points
            )


class TestModePerformanceThreshold:
    def test_creation(self, qpsk_ldpc_mode):
        model = ModePerformanceThreshold(
            modes=[qpsk_ldpc_mode],
            metric=ErrorMetric.FER,
            ebn0=DVB_S2_THRESHOLD_EBN0,
            error_rate=DVB_S2_THRESHOLD_ERROR_RATE,
            ref="ETSI EN 302 307-1",
        )

        assert_quantity_allclose(model.ebn0, DVB_S2_THRESHOLD_EBN0 * u.dB(1))
        assert_quantity_allclose(
            model.error_rate, DVB_S2_THRESHOLD_ERROR_RATE * u.dimensionless
        )
        assert model.metric == ErrorMetric.FER

    def test_threshold_properties(self, qpsk_ldpc_mode):
        """Test threshold properties return correct values with units."""
        model = ModePerformanceThreshold(
            modes=[qpsk_ldpc_mode],
            metric=ErrorMetric.FER,
            ebn0=DVB_S2_THRESHOLD_EBN0,
            error_rate=DVB_S2_THRESHOLD_ERROR_RATE,
        )

        assert_quantity_allclose(model.ebn0, DVB_S2_THRESHOLD_EBN0 * u.dB(1))
        assert_quantity_allclose(
            model.error_rate, DVB_S2_THRESHOLD_ERROR_RATE * u.dimensionless
        )

    def test_check(self, qpsk_ldpc_mode):
        """Test threshold checking for scalar and array inputs."""
        model = ModePerformanceThreshold(
            modes=[qpsk_ldpc_mode],
            metric=ErrorMetric.FER,
            ebn0=DVB_S2_THRESHOLD_EBN0,
            error_rate=DVB_S2_THRESHOLD_ERROR_RATE,
        )

        assert model.check(6.0 * u.dB(1)) is True
        assert model.check(5.0 * u.dB(1)) is True
        assert model.check(4.5 * u.dB(1)) is False

        ebn0_array = np.array([4.0, 5.0, 6.0]) * u.dB(1)
        expected = np.array([False, True, True])
        np.testing.assert_array_equal(model.check(ebn0_array), expected)

    def test_margin(self, qpsk_ldpc_mode):
        model = ModePerformanceThreshold(
            modes=[qpsk_ldpc_mode],
            metric=ErrorMetric.FER,
            ebn0=DVB_S2_THRESHOLD_EBN0,
            error_rate=DVB_S2_THRESHOLD_ERROR_RATE,
        )

        assert_quantity_allclose(model.margin(6.0 * u.dB(1)), 1.0 * u.dB(1))
        assert_quantity_allclose(model.margin(5.0 * u.dB(1)), 0.0 * u.dB(1))
        assert_quantity_allclose(model.margin(4.0 * u.dB(1)), -1.0 * u.dB(1))

        ebn0_array = np.array([4.0, 5.0, 6.0]) * u.dB(1)
        margins = model.margin(ebn0_array)
        assert_quantity_allclose(margins, np.array([-1.0, 0.0, 1.0]) * u.dB(1))

    def test_coding_gain_matching_error_rate(self, qpsk_ldpc_mode, qpsk_uncoded_mode):
        coded_model = ModePerformanceThreshold(
            modes=[qpsk_ldpc_mode],
            metric=ErrorMetric.FER,
            ebn0=DVB_S2_THRESHOLD_EBN0,
            error_rate=DVB_S2_THRESHOLD_ERROR_RATE,
        )

        uncoded_model = ModePerformanceThreshold(
            modes=[qpsk_uncoded_mode],
            metric=ErrorMetric.FER,
            ebn0=10.0,
            error_rate=DVB_S2_THRESHOLD_ERROR_RATE,
        )

        gain = coded_model.coding_gain(uncoded_model)
        assert_quantity_allclose(gain, 5.0 * u.dB(1))

    def test_coding_gain_mismatched_error_rate(self, qpsk_ldpc_mode):
        """Test that coding_gain raises ValueError when error rates differ."""
        threshold1 = ModePerformanceThreshold(
            modes=[qpsk_ldpc_mode],
            metric=ErrorMetric.FER,
            ebn0=5.0,
            error_rate=1e-7,
        )

        threshold2 = ModePerformanceThreshold(
            modes=[qpsk_ldpc_mode],
            metric=ErrorMetric.FER,
            ebn0=6.0,
            error_rate=1e-6,
        )

        with pytest.raises(ValueError):
            threshold1.coding_gain(threshold2)

    def test_coding_gain_mismatched_metric(self, qpsk_ldpc_mode):
        """Test that coding_gain raises ValueError when metrics differ."""
        fer_threshold = ModePerformanceThreshold(
            modes=[qpsk_ldpc_mode],
            metric=ErrorMetric.FER,
            ebn0=5.0,
            error_rate=1e-7,
        )

        ber_threshold = ModePerformanceThreshold(
            modes=[qpsk_ldpc_mode],
            metric=ErrorMetric.BER,
            ebn0=6.0,
            error_rate=1e-7,
        )

        with pytest.raises(ValueError):
            fer_threshold.coding_gain(ber_threshold)
