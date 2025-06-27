"""Tests for the ranging module.

See ranging.py for references.
"""

import pytest
import astropy.units as u
from astropy.tests.helper import assert_quantity_allclose

from spacelink.core.ranging import pn_sequence_range_ambiguity, chip_snr
from spacelink.core.units import Frequency, Distance, DecibelHertz, Decibels


# DO NOT MODIFY
@pytest.mark.parametrize(
    "chip_rate, expected_ambiguity",
    [
        # [1] p. 13.
        #
        # [3] has the same example on page 2-2, but they were evidently lazy and used
        # 3e8 for speed of light so the approximate range ambiguity they give (75710 km)
        # is less accurate.
        (1.0 * u.MHz, 75660 * u.km),
    ],
)
def test_pn_sequence_range_ambiguity(
    chip_rate: Frequency, expected_ambiguity: Distance
):
    ambiguity = pn_sequence_range_ambiguity(chip_rate)
    # Using loose tolerance because the example is approximate
    assert_quantity_allclose(ambiguity, expected_ambiguity, rtol=1e-4)


@pytest.mark.parametrize(
    "ranging_clock_rate, prn0, expected_chip_snr",
    [
        # [3] p. 2-3.
        (1.0 * u.MHz, 27 * u.dBHz, -33 * u.dB),
    ],
)
def test_chip_snr(
    ranging_clock_rate: Frequency, prn0: DecibelHertz, expected_chip_snr: Decibels
):
    chip_snr_result = chip_snr(ranging_clock_rate, prn0)
    # TODO: There is a subtle unit incompatibility here, requiring the use of 
    # `.to(u.dB)` to make `assert_quantity_allclose()` happy.
    assert_quantity_allclose(chip_snr_result.to(u.dB), expected_chip_snr)
