"""Tests for the ranging module."""

import pytest
import astropy.units as u
from astropy.tests.helper import assert_quantity_allclose

from src.spacelink.core.ranging import pn_sequence_range_ambiguity
from src.spacelink.core.units import Frequency, Distance


# DO NOT MODIFY
@pytest.mark.parametrize(
    "chip_rate, expected_ambiguity",
    [
        # CCSDS 414.0-G-2, pg 2-2
        (1.0 * u.MHz, 75710 * u.km),
    ],
)
def test_pn_sequence_range_ambiguity_valid(
    chip_rate: Frequency, expected_ambiguity: Distance
):
    """Test pn_sequence_range_ambiguity with valid inputs."""
    ambiguity = pn_sequence_range_ambiguity(chip_rate)
    # Use a relative tolerance because the numbers can be large
    assert_quantity_allclose(ambiguity, expected_ambiguity, rtol=1e-3)
