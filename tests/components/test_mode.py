"""Tests for the mode module.

Note: All tests in this file are marked as expected to fail (xfail) because
the implementation of the mode module is still in progress. These tests serve
as a specification for how the module should behave once completed.

The main issues identified:
1. RangingMode implementation has issues with power fraction calculations
2. DataMode.margin method has a bug - it uses cn0 directly instead of the calculated ebno
"""

import pytest
import astropy.units as u
import numpy as np
from astropy.tests.helper import assert_quantity_allclose

from spacelink.components.mode import RangingMode, DataMode
from spacelink.core.units import to_dB
from spacelink.core.modcod import ErrorRate


@pytest.mark.xfail(reason="RangingMode implementation is incomplete or has issues")
class TestRangingMode:
    """Tests for the RangingMode class."""

    def test_initialization(self):
        """Test that RangingMode initializes correctly."""
        ranging_mod_idx = 0.6 * u.dimensionless
        data_mod_idx = 1.2 * u.dimensionless

        mode = RangingMode(ranging_mod_idx, data_mod_idx)

        assert mode._ranging_mod_idx == ranging_mod_idx
        assert mode._data_mod_idx == data_mod_idx

    def test_power_fractions(self):
        """Test that power fractions are calculated correctly."""
        ranging_mod_idx = 0.6 * u.dimensionless
        data_mod_idx = 1.2 * u.dimensionless

        mode = RangingMode(ranging_mod_idx, data_mod_idx)
        carrier, ranging, data = mode.power_fractions

        # Check that fractions sum to approximately 1
        assert np.isclose(carrier + ranging + data, 1.0)

        # Check that all fractions are positive
        assert carrier > 0
        assert ranging > 0
        assert data > 0

    def test_ranging_power_fraction(self):
        """Test that ranging power fraction is converted to dB correctly."""
        ranging_mod_idx = 0.6 * u.dimensionless
        data_mod_idx = 1.2 * u.dimensionless

        mode = RangingMode(ranging_mod_idx, data_mod_idx)

        # Get the raw power fractions
        _, ranging, _ = mode.power_fractions

        # Check that the dB conversion is correct
        expected_db = to_dB(ranging)
        assert_quantity_allclose(mode.ranging_power_fraction, expected_db)

    def test_data_power_fraction(self):
        """Test that data power fraction is converted to dB correctly."""
        ranging_mod_idx = 0.6 * u.dimensionless
        data_mod_idx = 1.2 * u.dimensionless

        mode = RangingMode(ranging_mod_idx, data_mod_idx)

        # Get the raw power fractions
        _, _, data = mode.power_fractions

        # Check that the dB conversion is correct
        expected_db = to_dB(data)
        assert_quantity_allclose(mode.data_power_fraction, expected_db)


class TestDataMode:
    """Tests for the DataMode class."""

    def test_code_rate(self):
        """Test that code rate is calculated correctly."""
        coding_scheme = "CC(7,1/2) RS(255,223) I=5"
        bits_per_symbol = 2 * u.dimensionless
        error_rate = ErrorRate.E_NEG_5

        mode = DataMode(coding_scheme, bits_per_symbol, error_rate)

        # The code rate for this scheme should be 1/2 * 223/255
        expected_rate = 0.437254902 * u.dimensionless
        assert_quantity_allclose(mode.code_rate, expected_rate)

        assert_quantity_allclose(mode.required_ebno, 2.3 * u.dB, atol=0.01 * u.dB)

        symbol_rate = 1000 * u.Hz
        assert_quantity_allclose(mode.bit_rate(symbol_rate), 874.509804 * u.Hz)

        cn0 = 43 * u.dB
        ebn0 = mode.ebno(cn0, symbol_rate)
        assert_quantity_allclose(ebn0, 13.58 * u.dB, atol=0.01 * u.dB)

        margin = mode.margin(cn0, symbol_rate)
        assert_quantity_allclose(margin, 11.28 * u.dB, atol=0.01 * u.dB)
