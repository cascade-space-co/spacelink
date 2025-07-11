"""Tests for the ranging module.

See ranging.py for references.
"""

import io
import pytest
import astropy.units as u
from astropy.tests.helper import assert_quantity_allclose
from astropy.coordinates import Angle
import numpy as np
import pandas as pd

from spacelink.core import ranging
from spacelink.core.units import (
    Frequency,
    Distance,
    DecibelHertz,
    Decibels,
    Dimensionless,
    Time,
)


# DO NOT MODIFY
@pytest.mark.parametrize(
    "ranging_clock_rate, expected_ambiguity",
    [
        # [2] p. 13.
        #
        # [4] has the same example on page 2-2, but they were evidently lazy and used
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
        # [4] p. 2-3.
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
        (0.0 * u.rad, ranging.DataModulation.BIPOLAR, 1.0),
        (0.0 * u.rad, ranging.DataModulation.SINE_SUBCARRIER, 1.0),
        # Special cases for bipolar modulation.
        (np.pi / 4 * u.rad, ranging.DataModulation.BIPOLAR, 0.5),
        (np.pi / 2 * u.rad, ranging.DataModulation.BIPOLAR, 0.0),
        (np.pi * u.rad, ranging.DataModulation.BIPOLAR, 1.0),
        # Some hand-picked cases for sine subcarrier modulation computed using
        # WolframAlpha.
        (0.5 * u.rad, ranging.DataModulation.SINE_SUBCARRIER, 0.772382),
        (1.0 * u.rad, ranging.DataModulation.SINE_SUBCARRIER, 0.312631),
    ],
)
def test_suppression_factor(
    mod_idx: Angle,
    modulation: ranging.DataModulation,
    expected_suppression: Dimensionless,
):
    suppression = ranging._suppression_factor(mod_idx, modulation)
    assert_quantity_allclose(suppression, expected_suppression, atol=1e-10, rtol=1e-6)


@pytest.mark.parametrize(
    "mod_idx, modulation, expected_modulation_factor",
    [
        # No modulation expected when mod index is 0.
        (0.0 * u.rad, ranging.DataModulation.BIPOLAR, 0.0),
        (0.0 * u.rad, ranging.DataModulation.SINE_SUBCARRIER, 0.0),
        # Special cases for bipolar modulation.
        (np.pi / 4 * u.rad, ranging.DataModulation.BIPOLAR, 0.5),
        (np.pi / 2 * u.rad, ranging.DataModulation.BIPOLAR, 1.0),
        (np.pi * u.rad, ranging.DataModulation.BIPOLAR, 0.0),
        # Some hand-picked cases for sine subcarrier modulation computed using
        # WolframAlpha.
        (0.5 * u.rad, ranging.DataModulation.SINE_SUBCARRIER, 0.220331),
        (1.0 * u.rad, ranging.DataModulation.SINE_SUBCARRIER, 0.592879),
    ],
)
def test_modulation_factor(
    mod_idx: Angle,
    modulation: ranging.DataModulation,
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
    "mod_idx_ranging, mod_idx_data, modulation, expected_carrier_ratio",
    [
        # No modulation (all power in carrier)
        (0.0 * u.rad, 0.0 * u.rad, ranging.DataModulation.BIPOLAR, 1.0),
        # Ranging only
        (0.5 * u.rad, 0.0 * u.rad, ranging.DataModulation.BIPOLAR, 0.772382),
        # Data modulation only
        (0.0 * u.rad, np.pi / 4 * u.rad, ranging.DataModulation.BIPOLAR, 0.5),
        # Simultaneous ranging and data modulation, bipolar
        (
            0.5 * u.rad,
            np.pi / 4 * u.rad,
            ranging.DataModulation.BIPOLAR,
            0.5 * 0.772382,
        ),
        # Simultaneous ranging and data modulation, sine subcarrier
        (0.5 * u.rad, 0.5 * u.rad, ranging.DataModulation.SINE_SUBCARRIER, 0.772382**2),
    ],
)
def test_carrier_to_total_power(
    mod_idx_ranging: Angle,
    mod_idx_data: Angle,
    modulation: ranging.DataModulation,
    expected_carrier_ratio: Dimensionless,
):
    carrier_ratio = ranging.carrier_to_total_power(
        mod_idx_ranging, mod_idx_data, modulation
    )
    assert_quantity_allclose(carrier_ratio, expected_carrier_ratio, rtol=1e-4)


@pytest.mark.parametrize(
    "mod_idx_ranging, mod_idx_data, modulation, expected_ranging_ratio",
    [
        # No modulation (all power in carrier)
        (0.0 * u.rad, 0.0 * u.rad, ranging.DataModulation.BIPOLAR, 0.0),
        # Ranging only
        (0.5 * u.rad, 0.0 * u.rad, ranging.DataModulation.BIPOLAR, 0.220331),
        # Data modulation only
        (0.0 * u.rad, np.pi / 4 * u.rad, ranging.DataModulation.BIPOLAR, 0.0),
        # Simultaneous ranging and data modulation, bipolar
        (
            0.5 * u.rad,
            np.pi / 4 * u.rad,
            ranging.DataModulation.BIPOLAR,
            0.5 * 0.220331,
        ),
        # Simultaneous ranging and data modulation, sine subcarrier
        (0.5 * u.rad, 0.5 * u.rad, ranging.DataModulation.SINE_SUBCARRIER, 0.170180),
    ],
)
def test_ranging_to_total_power(
    mod_idx_ranging: Angle,
    mod_idx_data: Angle,
    modulation: ranging.DataModulation,
    expected_ranging_ratio: Dimensionless,
):
    ranging_ratio = ranging.ranging_to_total_power(
        mod_idx_ranging, mod_idx_data, modulation
    )
    assert_quantity_allclose(ranging_ratio, expected_ranging_ratio, rtol=1e-4)


@pytest.mark.parametrize(
    "mod_idx_ranging, mod_idx_data, modulation, expected_data_ratio",
    [
        # No modulation (all power in carrier)
        (0.0 * u.rad, 0.0 * u.rad, ranging.DataModulation.BIPOLAR, 0.0),
        # Ranging only
        (0.5 * u.rad, 0.0 * u.rad, ranging.DataModulation.BIPOLAR, 0.0),
        # Data modulation only
        (0.0 * u.rad, np.pi / 4 * u.rad, ranging.DataModulation.BIPOLAR, 0.5),
        # Simultaneous ranging and data modulation, bipolar
        (
            0.5 * u.rad,
            np.pi / 4 * u.rad,
            ranging.DataModulation.BIPOLAR,
            0.5 * 0.772382,
        ),
        # Simultaneous ranging and data modulation, sine subcarrier
        (0.5 * u.rad, 0.5 * u.rad, ranging.DataModulation.SINE_SUBCARRIER, 0.170180),
    ],
)
def test_data_to_total_power(
    mod_idx_ranging: Angle,
    mod_idx_data: Angle,
    modulation: ranging.DataModulation,
    expected_data_ratio: Dimensionless,
):
    data_ratio = ranging.data_to_total_power(mod_idx_ranging, mod_idx_data, modulation)
    assert_quantity_allclose(data_ratio, expected_data_ratio, rtol=1e-4)


def generate_pn_acquisition_test_params():
    """Generate test parameters for pn_component_acquisition_probability."""

    # Raw values copied directly from [2] Table 6.
    dsn_table_6_raw = [
        "-0.050 5.7 6.5 6.9 7.1 7.4",
        "-0.040 6.2 6.9 7.2 7.5 7.7",
        "-0.030 6.7 7.3 7.7 7.9 8.1",
        "-0.020 7.4 7.9 8.3 8.5 8.7",
        "-0.010 8.3 8.8 9.1 9.3 9.4",
        "-0.009 8.4 8.9 9.2 9.4 9.5",
        "-0.008 8.6 9.0 9.3 9.5 9.7",
        "-0.007 8.7 9.2 9.4 9.6 9.8",
        "-0.006 8.9 9.3 9.6 9.8 9.9",
        "-0.005 9.1 9.5 9.8 9.9 10.1",
        "-0.004 9.3 9.7 10.0 10.1 10.3",
        "-0.003 9.6 10.0 10.2 10.4 10.5",
        "-0.002 9.9 10.3 10.5 10.7 10.8",
        "-0.001 10.5 10.8 11.0 11.1 11.3",
    ]
    df = pd.read_csv(io.StringIO("\n".join(dsn_table_6_raw)), sep=" ", header=None)
    df.columns = ["logPn", 2, 3, 4, 5, 6]

    params = []
    for code in ranging.PnRangingCode:
        for _, row in df.iterrows():
            for component in [2, 3, 4, 5, 6]:
                corr_coeff = ranging._CORR_COEFF_DSN[code][component]
                ranging_to_noise_psd = (
                    10 ** (row[component] / 10) / corr_coeff**2 * u.Hz
                )
                integration_time = 1.0 * u.s
                expected_probability = 10 ** row["logPn"]

                params.append(
                    (
                        ranging_to_noise_psd,
                        integration_time,
                        code,
                        component,
                        expected_probability,
                        f"{code=}, {component=}, {expected_probability=}, "
                        f"{ranging_to_noise_psd=}",
                    )
                )
    return params


@pytest.mark.parametrize(
    "ranging_to_noise_psd, integration_time, code, component, expected_probability, "
    "case_description",
    generate_pn_acquisition_test_params(),
)
def test_pn_component_acquisition_probability(
    ranging_to_noise_psd: Frequency,
    integration_time: Time,
    code: ranging.PnRangingCode,
    component: int,
    expected_probability: float,
    case_description: str,
):
    """Test pn_component_acquisition_probability using Table 6 data from [2]."""
    result = ranging.pn_component_acquisition_probability(
        ranging_to_noise_psd, integration_time, code, component
    )

    assert_quantity_allclose(
        result,
        expected_probability,
        rtol=5e-3,  # [2] Table 6 has only 3 significant digits for log(Pn)
        err_msg=f"Failed for case: {case_description}",
    )


def generate_pn_acquisition_full_test_params():
    """Generate test parameters for pn_acquisition_probability using DSN plot data."""

    # In the following dicts keys are T*Pr/N0 in dB and values are acquisition probability.
    acq_prob_data_from_dsn_plots = {
        # Extracted from [2] Figure 14 using WebPlotDigitizer 4.7
        ranging.PnRangingCode.DSN: {
            30.0: 0.10025,
            31.0: 0.17544,
            32.0: 0.29449,
            33.0: 0.45238,
            34.0: 0.62907,
            35.0: 0.78822,
            36.0: 0.90351,
            37.0: 0.96617,
            38.0: 0.99248,
        },
        # Extracted from [2] Figure 14 using WebPlotDigitizer 4.7
        ranging.PnRangingCode.CCSDS_T4B: {
            30.0: 0.38095,
            31.0: 0.55138,
            32.0: 0.72306,
            33.0: 0.86090,
            34.0: 0.94486,
            35.0: 0.98371,
            36.0: 0.99499,
            37.0: 1.00000,
            38.0: 1.00000,
        },
        # Extracted from [2] Figure 15 using WebPlotDigitizer 4.7
        ranging.PnRangingCode.CCSDS_T2B: {
            16.0: 0.15233,
            17.0: 0.25799,
            18.0: 0.40541,
            19.0: 0.57862,
            20.0: 0.74816,
            21.0: 0.87838,
            22.0: 0.95332,
            23.0: 0.98771,
            24.0: 0.99754,
        },
    }

    params = []
    for code, data in acq_prob_data_from_dsn_plots.items():
        for snr_db, expected_probability in data.items():
            params.append(
                (
                    snr_db * u.dBHz,
                    1.0 * u.s,
                    code,
                    expected_probability,
                    f"{code=}, T*Pr/N0={snr_db} dB, P_acq={expected_probability}",
                )
            )

    return params


@pytest.mark.parametrize(
    "ranging_to_noise_psd, integration_time, code, expected_probability, case_description",
    generate_pn_acquisition_full_test_params(),
)
def test_pn_acquisition_probability(
    ranging_to_noise_psd: DecibelHertz,
    integration_time: Time,
    code: ranging.PnRangingCode,
    expected_probability: float,
    case_description: str,
):
    """Test pn_acquisition_probability using data extracted from DSN plots."""
    result = ranging.pn_acquisition_probability(
        ranging_to_noise_psd, integration_time, code
    )

    assert_quantity_allclose(
        result,
        expected_probability,
        atol=5e-3,  # Test values are from plot extraction
        err_msg=f"Failed for case: {case_description}",
    )
