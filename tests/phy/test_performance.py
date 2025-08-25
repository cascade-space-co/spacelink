from fractions import Fraction

import astropy.units as u
import numpy as np
import pytest

from spacelink.phy.mode import Code, CodeChain, LinkMode, Modulation
from spacelink.phy.performance import (
    ErrorMetric,
    ModePerformance,
)


class TestModePerformance:
    def test_creation_with_points(self):
        """Test creating ModePerformance with points list."""
        points = [(0.0, 1e-1), (1.0, 3e-2), (2.0, 5e-3), (3.0, 1e-4)]

        # Create a minimal LinkMode for testing
        modulation = Modulation(name="BPSK", bits_per_symbol=1)
        coding = CodeChain(codes=[Code(name="uncoded", rate=Fraction(1))])
        mode = LinkMode(id="TEST", modulation=modulation, coding=coding)

        model = ModePerformance(
            modes=[mode],
            metric=ErrorMetric.BER,
            points=points,
        )

        assert model.points == points

    def test_ber_interpolation(self):
        """Test BER interpolation functionality."""
        points = [(0.0, 1e-1), (1.0, 3e-2), (2.0, 5e-3), (3.0, 1e-4)]

        # Create a minimal LinkMode for testing
        modulation = Modulation(name="BPSK", bits_per_symbol=1)
        coding = CodeChain(codes=[Code(name="uncoded", rate=Fraction(1))])
        mode = LinkMode(id="TEST", modulation=modulation, coding=coding)

        model = ModePerformance(
            modes=[mode],
            metric=ErrorMetric.BER,
            points=points,
        )

        # Test exact points
        assert model.ebn0_to_error_rate(0.0 * u.dB).value == pytest.approx(1e-1)
        assert model.ebn0_to_error_rate(1.0 * u.dB).value == pytest.approx(3e-2)
        assert model.ebn0_to_error_rate(2.0 * u.dB).value == pytest.approx(5e-3)
        assert model.ebn0_to_error_rate(3.0 * u.dB).value == pytest.approx(1e-4)

        # Test interpolation
        ber_1_5 = model.ebn0_to_error_rate(1.5 * u.dB).value
        assert 5e-3 < ber_1_5 < 3e-2  # Should be between the values at 1.0 and 2.0

        # Test reverse interpolation (exact points)
        assert model.error_rate_to_ebn0(1e-1 * u.dimensionless).value == pytest.approx(
            0.0
        )
        assert model.error_rate_to_ebn0(3e-2 * u.dimensionless).value == pytest.approx(
            1.0
        )
        assert model.error_rate_to_ebn0(5e-3 * u.dimensionless).value == pytest.approx(
            2.0
        )
        assert model.error_rate_to_ebn0(1e-4 * u.dimensionless).value == pytest.approx(
            3.0
        )

        # Test reverse interpolation (interpolated value)
        ebn0_1_5 = model.error_rate_to_ebn0(ber_1_5 * u.dimensionless).value
        assert 1.0 < ebn0_1_5 < 2.0  # Should be between the values at 1.0 and 2.0

        # Test array arguments for ebn0_to_error_rate
        ebn0_array = np.array([0.0, 1.0, 2.0, 3.0]) * u.dB
        error_rates = model.ebn0_to_error_rate(ebn0_array)
        np.testing.assert_allclose(error_rates.value, [1e-1, 3e-2, 5e-3, 1e-4])

        # Test array arguments for error_rate_to_ebn0
        error_rate_array = np.array([1e-1, 3e-2, 5e-3, 1e-4]) * u.dimensionless
        ebn0_values = model.error_rate_to_ebn0(error_rate_array)
        np.testing.assert_allclose(ebn0_values.value, [0.0, 1.0, 2.0, 3.0], atol=1e-15)

    def test_coding_gain(self):
        """Test coding gain calculation functionality."""
        modulation = Modulation(name="BPSK", bits_per_symbol=1)
        coding = CodeChain(codes=[Code(name="uncoded", rate=Fraction(1))])
        mode = LinkMode(id="TEST", modulation=modulation, coding=coding)

        # Create a "coded" model with better performance
        coded_points = [(0.0, 1e-0), (1.0, 1e-1), (2.0, 1e-2), (3.0, 1e-3)]
        coded_model = ModePerformance(
            modes=[mode],
            metric=ErrorMetric.BER,
            points=coded_points,
        )

        # Create an "uncoded" model with worse performance (offset by 1, 2, 3, 4 dB)
        uncoded_points = [(1.0, 1e-0), (3.0, 1e-1), (5.0, 1e-2), (7.0, 1e-3)]
        uncoded_model = ModePerformance(
            modes=[mode],
            metric=ErrorMetric.BER,
            points=uncoded_points,
        )

        gain_1e0 = coded_model.coding_gain(uncoded_model, 1e-0 * u.dimensionless)
        assert gain_1e0.value == pytest.approx(1.0)

        gain_1e1 = coded_model.coding_gain(uncoded_model, 1e-1 * u.dimensionless)
        assert gain_1e1.value == pytest.approx(2.0)

        gain_1e2 = coded_model.coding_gain(uncoded_model, 1e-2 * u.dimensionless)
        assert gain_1e2.value == pytest.approx(3.0)

        # Test array arguments for coding_gain
        error_rate_array = np.array([1e-0, 1e-1, 1e-2]) * u.dimensionless
        gains = coded_model.coding_gain(uncoded_model, error_rate_array)
        np.testing.assert_allclose(gains, np.array([1.0, 2.0, 3.0]) * u.dB)

    def test_edge_cases(self):
        """Test edge cases for interpolation."""
        points = [(0.0, 1e-1), (1.0, 3e-2), (2.0, 5e-3), (3.0, 1e-4)]

        # Create a minimal LinkMode for testing
        modulation = Modulation(name="BPSK", bits_per_symbol=1)
        coding = CodeChain(codes=[Code(name="uncoded", rate=Fraction(1))])
        mode = LinkMode(id="TEST", modulation=modulation, coding=coding)

        model = ModePerformance(
            modes=[mode],
            metric=ErrorMetric.BER,
            points=points,
        )

        # Test values beyond range - should return NaN
        assert np.isnan(model.ebn0_to_error_rate(-1.0 * u.dB).value)
        assert np.isnan(model.ebn0_to_error_rate(4.0 * u.dB).value)
        assert np.isnan(model.error_rate_to_ebn0(1e-0 * u.dimensionless).value)
        assert np.isnan(model.error_rate_to_ebn0(1e-5 * u.dimensionless).value)

    def test_insufficient_points(self):
        """Test behavior with insufficient points for interpolation."""
        # Create a minimal LinkMode for testing
        modulation = Modulation(name="BPSK", bits_per_symbol=1)
        coding = CodeChain(codes=[Code(name="uncoded", rate=Fraction(1))])
        mode = LinkMode(id="TEST", modulation=modulation, coding=coding)

        # Creating ModePerformance with empty points should raise an exception
        with pytest.raises(ValueError):
            ModePerformance(
                modes=[mode],
                metric=ErrorMetric.BER,
                points=[],
            )

        # Creating ModePerformance with single point should also raise an exception
        with pytest.raises(ValueError):
            ModePerformance(
                modes=[mode],
                metric=ErrorMetric.BER,
                points=[(1.0, 1e-2)],
            )

    def test_multiple_modes(self):
        """Test creating ModePerformance with multiple modes."""
        points = [(0.0, 1e-1), (1.0, 3e-2), (2.0, 5e-3), (3.0, 1e-4)]

        # Create multiple LinkModes for testing
        modulation1 = Modulation(name="BPSK", bits_per_symbol=1)
        coding1 = CodeChain(codes=[Code(name="uncoded", rate=Fraction(1))])
        mode1 = LinkMode(id="TEST1", modulation=modulation1, coding=coding1)

        modulation2 = Modulation(name="QPSK", bits_per_symbol=2)
        coding2 = CodeChain(codes=[Code(name="uncoded", rate=Fraction(1))])
        mode2 = LinkMode(id="TEST2", modulation=modulation2, coding=coding2)

        model = ModePerformance(
            modes=[mode1, mode2],
            metric=ErrorMetric.BER,
            points=points,
        )

        assert len(model.modes) == 2
        assert model.modes[0].id == "TEST1"
        assert model.modes[1].id == "TEST2"

    def test_coding_gain_metric_mismatch_error(self):
        """Test that coding_gain raises ValueError when metrics don't match."""
        modulation = Modulation(name="BPSK", bits_per_symbol=1)
        coding = CodeChain(codes=[Code(name="uncoded", rate=Fraction(1))])
        mode = LinkMode(id="TEST", modulation=modulation, coding=coding)

        points = [(0.0, 1e-1), (1.0, 3e-2), (2.0, 5e-3), (3.0, 1e-4)]

        # Create two models with different error metrics
        ber_model = ModePerformance(
            modes=[mode],
            metric=ErrorMetric.BER,
            points=points,
        )

        fer_model = ModePerformance(
            modes=[mode],
            metric=ErrorMetric.FER,
            points=points,
        )

        # Should raise ValueError when trying to compare different metrics
        with pytest.raises(ValueError):
            ber_model.coding_gain(fer_model, 1e-2 * u.dimensionless)

    def test_unsorted_points_validation(self):
        """Test that unsorted points are rejected during construction."""
        modulation = Modulation(name="BPSK", bits_per_symbol=1)
        coding = CodeChain(codes=[Code(name="uncoded", rate=Fraction(1))])
        mode = LinkMode(id="TEST", modulation=modulation, coding=coding)

        # Test decreasing order
        unsorted_points_decreasing = [
            (3.0, 1e-4),
            (2.0, 5e-3),
            (1.0, 3e-2),
            (0.0, 1e-1),
        ]
        with pytest.raises(ValueError):
            ModePerformance(
                modes=[mode],
                metric=ErrorMetric.BER,
                points=unsorted_points_decreasing,
            )

        # Test mixed order
        unsorted_points_mixed = [(0.0, 1e-1), (2.0, 5e-3), (1.0, 3e-2), (3.0, 1e-4)]
        with pytest.raises(ValueError):
            ModePerformance(
                modes=[mode],
                metric=ErrorMetric.BER,
                points=unsorted_points_mixed,
            )

        # Test equal Eb/N0 values
        duplicate_ebn0_points = [(0.0, 1e-1), (1.0, 3e-2), (1.0, 5e-3), (3.0, 1e-4)]
        with pytest.raises(ValueError):
            ModePerformance(
                modes=[mode],
                metric=ErrorMetric.BER,
                points=duplicate_ebn0_points,
            )

        # Test that properly sorted points still work
        sorted_points = [(0.0, 1e-1), (1.0, 3e-2), (2.0, 5e-3), (3.0, 1e-4)]
        model = ModePerformance(
            modes=[mode],
            metric=ErrorMetric.BER,
            points=sorted_points,
        )
        assert model.points == sorted_points

    def test_non_decreasing_error_values_validation(self):
        """Test that non-decreasing error values are rejected during construction."""
        modulation = Modulation(name="BPSK", bits_per_symbol=1)
        coding = CodeChain(codes=[Code(name="uncoded", rate=Fraction(1))])
        mode = LinkMode(id="TEST", modulation=modulation, coding=coding)

        # Test increasing error values (should be decreasing)
        increasing_error_points = [(0.0, 1e-4), (1.0, 3e-3), (2.0, 5e-2), (3.0, 1e-1)]
        with pytest.raises(ValueError):
            ModePerformance(
                modes=[mode],
                metric=ErrorMetric.BER,
                points=increasing_error_points,
            )

        # Test constant error values
        constant_error_points = [(0.0, 1e-2), (1.0, 1e-2), (2.0, 1e-2), (3.0, 1e-2)]
        with pytest.raises(ValueError):
            ModePerformance(
                modes=[mode],
                metric=ErrorMetric.BER,
                points=constant_error_points,
            )

        # Test mixed non-monotonic error values
        mixed_error_points = [(0.0, 1e-1), (1.0, 3e-2), (2.0, 5e-2), (3.0, 1e-4)]
        with pytest.raises(ValueError):
            ModePerformance(
                modes=[mode],
                metric=ErrorMetric.BER,
                points=mixed_error_points,
            )
