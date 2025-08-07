import pytest
import numpy as np
import astropy.units as u
from fractions import Fraction
from spacelink.phy.performance import (
    ModePerformance,
    ErrorMetric,
)
from spacelink.phy.mode import LinkMode, Modulation, CodeChain, Code


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
        assert model.ebno_to_error_rate(0.0 * u.dB).value == pytest.approx(1e-1)
        assert model.ebno_to_error_rate(1.0 * u.dB).value == pytest.approx(3e-2)
        assert model.ebno_to_error_rate(2.0 * u.dB).value == pytest.approx(5e-3)
        assert model.ebno_to_error_rate(3.0 * u.dB).value == pytest.approx(1e-4)

        # Test interpolation
        ber_1_5 = model.ebno_to_error_rate(1.5 * u.dB).value
        assert 5e-3 < ber_1_5 < 3e-2  # Should be between the values at 1.0 and 2.0

        # Test reverse interpolation (exact points)
        assert model.error_rate_to_ebno(1e-1 * u.dimensionless).value == pytest.approx(
            0.0
        )
        assert model.error_rate_to_ebno(3e-2 * u.dimensionless).value == pytest.approx(
            1.0
        )
        assert model.error_rate_to_ebno(5e-3 * u.dimensionless).value == pytest.approx(
            2.0
        )
        assert model.error_rate_to_ebno(1e-4 * u.dimensionless).value == pytest.approx(
            3.0
        )

        # Test reverse interpolation (interpolated value)
        ebno_1_5 = model.error_rate_to_ebno(ber_1_5 * u.dimensionless).value
        assert 1.0 < ebno_1_5 < 2.0  # Should be between the values at 1.0 and 2.0

        # Test array arguments for ebno_to_error_rate
        ebno_array = np.array([0.0, 1.0, 2.0, 3.0]) * u.dB
        error_rates = model.ebno_to_error_rate(ebno_array)
        np.testing.assert_allclose(error_rates.value, [1e-1, 3e-2, 5e-3, 1e-4])

        # Test array arguments for error_rate_to_ebno
        error_rate_array = np.array([1e-1, 3e-2, 5e-3, 1e-4]) * u.dimensionless
        ebno_values = model.error_rate_to_ebno(error_rate_array)
        np.testing.assert_allclose(ebno_values.value, [0.0, 1.0, 2.0, 3.0], atol=1e-15)

    def test_coding_gain(self):
        """Test coding gain calculation functionality."""
        modulation = Modulation(name="BPSK", bits_per_symbol=1)
        coding = CodeChain(codes=[Code(name="uncoded", rate=Fraction(1))])
        mode = LinkMode(id="TEST", modulation=modulation, coding=coding)

        # Create a "coded" model with better performance
        coded_points = [(0.0, 1e-3), (1.0, 1e-2), (2.0, 1e-1), (3.0, 1e-0)]
        coded_model = ModePerformance(
            modes=[mode],
            metric=ErrorMetric.BER,
            points=coded_points,
        )

        # Create an "uncoded" model with worse performance (offset by 1, 2, 3, 4 dB)
        uncoded_points = [(1.0, 1e-3), (3.0, 1e-2), (5.0, 1e-1), (7.0, 1e-0)]
        uncoded_model = ModePerformance(
            modes=[mode],
            metric=ErrorMetric.BER,
            points=uncoded_points,
        )

        gain_1e3 = coded_model.coding_gain(uncoded_model, 1e-3 * u.dimensionless)
        assert gain_1e3.value == pytest.approx(1.0)

        gain_1e2 = coded_model.coding_gain(uncoded_model, 1e-2 * u.dimensionless)
        assert gain_1e2.value == pytest.approx(2.0)

        gain_1e1 = coded_model.coding_gain(uncoded_model, 1e-1 * u.dimensionless)
        assert gain_1e1.value == pytest.approx(3.0)

        # Test array arguments for coding_gain
        error_rate_array = np.array([1e-3, 1e-2, 1e-1]) * u.dimensionless
        gains = coded_model.coding_gain(uncoded_model, error_rate_array)
        np.testing.assert_allclose(gains.value, [1.0, 2.0, 3.0])

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
        assert np.isnan(model.ebno_to_error_rate(-1.0 * u.dB).value)
        assert np.isnan(model.ebno_to_error_rate(4.0 * u.dB).value)
        assert np.isnan(model.error_rate_to_ebno(1e-0 * u.dimensionless).value)
        assert np.isnan(model.error_rate_to_ebno(1e-5 * u.dimensionless).value)

    def test_empty_model(self):
        """Test behavior with empty points."""
        # Create a minimal LinkMode for testing
        modulation = Modulation(name="BPSK", bits_per_symbol=1)
        coding = CodeChain(codes=[Code(name="uncoded", rate=Fraction(1))])
        mode = LinkMode(id="TEST", modulation=modulation, coding=coding)

        # Creating ModePerformance with empty points should raise an exception
        with pytest.raises(IndexError):
            model = ModePerformance(
                modes=[mode],
                metric=ErrorMetric.BER,
                points=[],
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
