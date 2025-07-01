"""Tests for the ranging module.

See ranging.py for references.
"""

import pytest
import astropy.units as u
from astropy.tests.helper import assert_quantity_allclose
from astropy.coordinates import Angle
import numpy as np

from spacelink.core import ranging
from spacelink.core.units import (
    Frequency,
    Distance,
    DecibelHertz,
    Decibels,
    Dimensionless,
)


# DO NOT MODIFY
@pytest.mark.parametrize(
    "ranging_clock_rate, expected_ambiguity",
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
    ranging_clock_rate: Frequency, expected_ambiguity: Distance
):
    ambiguity = ranging.pn_sequence_range_ambiguity(ranging_clock_rate)
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
    chip_snr_result = ranging.chip_snr(ranging_clock_rate, prn0)
    # TODO: There is a subtle unit incompatibility here, requiring the use of
    # `.to(u.dB)` to make `assert_quantity_allclose()` happy.
    assert_quantity_allclose(chip_snr_result.to(u.dB), expected_chip_snr)


@pytest.mark.parametrize(
    "mod_idx, modulation, expected_suppression",
    [
        # No suppression expected when mod index is 0.
        (0.0 * u.rad, ranging.CommandMod.BIPOLAR, 1.0),
        (0.0 * u.rad, ranging.CommandMod.SINE_SUBCARRIER, 1.0),
        # Special cases for bipolar modulation.
        (np.pi / 4 * u.rad, ranging.CommandMod.BIPOLAR, 0.5),
        (np.pi / 2 * u.rad, ranging.CommandMod.BIPOLAR, 0.0),
        (np.pi * u.rad, ranging.CommandMod.BIPOLAR, 1.0),
        # Some hand-picked cases for sine subcarrier modulation computed using
        # WolframAlpha.
        (0.5 * u.rad, ranging.CommandMod.SINE_SUBCARRIER, 0.772382),
        (1.0 * u.rad, ranging.CommandMod.SINE_SUBCARRIER, 0.312631),
    ],
)
def test_suppression_factor(
    mod_idx: Angle, modulation: ranging.CommandMod, expected_suppression: Dimensionless
):
    suppression = ranging._suppression_factor(mod_idx, modulation)
    assert_quantity_allclose(suppression, expected_suppression, atol=1e-10, rtol=1e-6)


@pytest.mark.parametrize(
    "mod_idx, modulation, expected_modulation_factor",
    [
        # No modulation expected when mod index is 0.
        (0.0 * u.rad, ranging.CommandMod.BIPOLAR, 0.0),
        (0.0 * u.rad, ranging.CommandMod.SINE_SUBCARRIER, 0.0),
        # Special cases for bipolar modulation.
        (np.pi / 4 * u.rad, ranging.CommandMod.BIPOLAR, 0.5),
        (np.pi / 2 * u.rad, ranging.CommandMod.BIPOLAR, 1.0),
        (np.pi * u.rad, ranging.CommandMod.BIPOLAR, 0.0),
        # Some hand-picked cases for sine subcarrier modulation computed using
        # WolframAlpha.
        (0.5 * u.rad, ranging.CommandMod.SINE_SUBCARRIER, 0.220331),
        (1.0 * u.rad, ranging.CommandMod.SINE_SUBCARRIER, 0.592879),
    ],
)
def test_modulation_factor(
    mod_idx: Angle,
    modulation: ranging.CommandMod,
    expected_modulation_factor: Dimensionless,
):
    modulation_factor = ranging._modulation_factor(mod_idx, modulation)
    assert_quantity_allclose(
        modulation_factor, expected_modulation_factor, atol=1e-10, rtol=1e-6
    )


def test_invalid_modulation():
    """Test that ValueError is raised for invalid modulation type."""
    invalid_modulation = "hogwash"
    with pytest.raises(ValueError):
        ranging._modulation_factor(0.5 * u.rad, invalid_modulation)
    with pytest.raises(ValueError):
        ranging._suppression_factor(0.5 * u.rad, invalid_modulation)


@pytest.mark.parametrize(
    "mod_idx_ranging, mod_idx_cmd, modulation, expected_carrier_ratio",
    [
        # No modulation (all power in carrier)
        (0.0 * u.rad, 0.0 * u.rad, ranging.CommandMod.BIPOLAR, 1.0),
        # Ranging only
        (0.5 * u.rad, 0.0 * u.rad, ranging.CommandMod.BIPOLAR, 0.772382),
        # Data modulation only
        (0.0 * u.rad, np.pi / 4 * u.rad, ranging.CommandMod.BIPOLAR, 0.5),
        # Simultaneous ranging and data modulation, bipolar
        (0.5 * u.rad, np.pi / 4 * u.rad, ranging.CommandMod.BIPOLAR, 0.5 * 0.772382),
        # Simultaneous ranging and data modulation, sine subcarrier
        (0.5 * u.rad, 0.5 * u.rad, ranging.CommandMod.SINE_SUBCARRIER, 0.772382**2),
    ],
)
def test_uplink_carrier_to_total_power(
    mod_idx_ranging: Angle,
    mod_idx_cmd: Angle,
    modulation: ranging.CommandMod,
    expected_carrier_ratio: Dimensionless,
):
    carrier_ratio = ranging.uplink_carrier_to_total_power(
        mod_idx_ranging, mod_idx_cmd, modulation
    )
    assert_quantity_allclose(carrier_ratio, expected_carrier_ratio, rtol=1e-4)


@pytest.mark.parametrize(
    "mod_idx_ranging, mod_idx_cmd, modulation, expected_ranging_ratio",
    [
        # No modulation (all power in carrier)
        (0.0 * u.rad, 0.0 * u.rad, ranging.CommandMod.BIPOLAR, 0.0),
        # Ranging only
        (0.5 * u.rad, 0.0 * u.rad, ranging.CommandMod.BIPOLAR, 0.220331),
        # Data modulation only
        (0.0 * u.rad, np.pi / 4 * u.rad, ranging.CommandMod.BIPOLAR, 0.0),
        # Simultaneous ranging and data modulation, bipolar
        (0.5 * u.rad, np.pi / 4 * u.rad, ranging.CommandMod.BIPOLAR, 0.5 * 0.220331),
        # Simultaneous ranging and data modulation, sine subcarrier
        (0.5 * u.rad, 0.5 * u.rad, ranging.CommandMod.SINE_SUBCARRIER, 0.170180),
    ],
)
def test_uplink_ranging_to_total_power(
    mod_idx_ranging: Angle,
    mod_idx_cmd: Angle,
    modulation: ranging.CommandMod,
    expected_ranging_ratio: Dimensionless,
):
    ranging_ratio = ranging.uplink_ranging_to_total_power(
        mod_idx_ranging, mod_idx_cmd, modulation
    )
    assert_quantity_allclose(ranging_ratio, expected_ranging_ratio, rtol=1e-4)


@pytest.mark.parametrize(
    "mod_idx_ranging, mod_idx_cmd, modulation, expected_data_ratio",
    [
        # No modulation (all power in carrier)
        (0.0 * u.rad, 0.0 * u.rad, ranging.CommandMod.BIPOLAR, 0.0),
        # Ranging only
        (0.5 * u.rad, 0.0 * u.rad, ranging.CommandMod.BIPOLAR, 0.0),
        # Data modulation only
        (0.0 * u.rad, np.pi / 4 * u.rad, ranging.CommandMod.BIPOLAR, 0.5),
        # Simultaneous ranging and data modulation, bipolar
        (0.5 * u.rad, np.pi / 4 * u.rad, ranging.CommandMod.BIPOLAR, 0.5 * 0.772382),
        # Simultaneous ranging and data modulation, sine subcarrier
        (0.5 * u.rad, 0.5 * u.rad, ranging.CommandMod.SINE_SUBCARRIER, 0.170180),
    ],
)
def test_uplink_data_to_total_power(
    mod_idx_ranging: Angle,
    mod_idx_cmd: Angle,
    modulation: ranging.CommandMod,
    expected_data_ratio: Dimensionless,
):
    data_ratio = ranging.uplink_data_to_total_power(
        mod_idx_ranging, mod_idx_cmd, modulation
    )
    assert_quantity_allclose(data_ratio, expected_data_ratio, rtol=1e-4)
