"""Tests for the ranging module.

See ranging.py for references.
"""

import math

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import Angle
from astropy.tests.helper import assert_quantity_allclose

from spacelink.core import ranging
from spacelink.core.units import (
    DecibelHertz,
    Decibels,
    Dimensionless,
    Distance,
    Frequency,
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


def test_pn_acquisition_time_invalid_inputs():
    # success_probability out of bounds
    with pytest.raises(ValueError):
        ranging.pn_acquisition_time(
            30.0 * u.dBHz, 0.0 * u.dimensionless, ranging.PnRangingCode.DSN
        )
    with pytest.raises(ValueError):
        ranging.pn_acquisition_time(
            30.0 * u.dBHz, 1.0 * u.dimensionless, ranging.PnRangingCode.DSN
        )

    # invalid code
    class _BadCode:
        pass

    with pytest.raises(ValueError):
        ranging.pn_acquisition_time(30.0 * u.dBHz, 0.5 * u.dimensionless, _BadCode())


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


def gen_pn_component_acq_prob_test_params():
    """Generate test parameters for pn_component_acquisition_probability."""

    # Raw values copied directly from [2] Table 6.
    # First column is log10(Pn).
    # Remaining columns are (Rn**2 * T * Pr/N0) in dB for components 2-6.
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

    params = []
    integration_time = 1.0 * u.s
    components = [2, 3, 4, 5, 6]
    for code in ranging.PnRangingCode:
        for row in dsn_table_6_raw:
            row_values = [float(v) for v in row.split()]
            expected_probability = 10 ** row_values[0]
            for component, table_val in zip(components, row_values[1:], strict=False):
                corr_coeff = ranging._CORR_COEFF_DSN[code][component]
                ranging_to_noise_psd = 10 ** (table_val / 10) / corr_coeff**2 * u.Hz

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
    gen_pn_component_acq_prob_test_params(),
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


# In the following dicts keys are T*Pr/N0 in dB and values are acquisition
# probability.
ACQ_PROB_DATA_FROM_DSN_PLOTS = {
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


def gen_pn_acq_prob_test_params():
    params = []
    for code, data in ACQ_PROB_DATA_FROM_DSN_PLOTS.items():
        for snr_db, expected_probability in data.items():
            params.append(
                (
                    snr_db * u.dBHz,
                    1.0 * u.s,
                    code,
                    expected_probability * u.dimensionless,
                    f"{code=}, T*Pr/N0={snr_db} dB, P_acq={expected_probability}",
                )
            )

    return params


@pytest.mark.parametrize(
    "ranging_to_noise_psd, integration_time, code, expected_probability, "
    "case_description",
    gen_pn_acq_prob_test_params(),
)
def test_pn_acquisition_probability(
    ranging_to_noise_psd: DecibelHertz,
    integration_time: Time,
    code: ranging.PnRangingCode,
    expected_probability: Dimensionless,
    case_description: str,
):
    result = ranging.pn_acquisition_probability(
        ranging_to_noise_psd, integration_time, code
    )

    assert_quantity_allclose(
        result,
        expected_probability,
        atol=5e-3,  # Test values are from plot extraction
        err_msg=f"Failed for case: {case_description}",
    )


def gen_pn_acq_time_test_params():
    params = []
    ranging_to_noise_psd = 30 * u.dBHz
    for code, data in ACQ_PROB_DATA_FROM_DSN_PLOTS.items():
        for snr_db, success_probability in data.items():
            if not 0 < success_probability < 1:
                continue
            params.append(
                (
                    ranging_to_noise_psd,
                    success_probability * u.dimensionless,
                    code,
                    snr_db * u.dB - ranging_to_noise_psd,
                    f"{code=}, T*Pr/N0={snr_db} dB, P_acq={success_probability}",
                )
            )

    return params


@pytest.mark.parametrize(
    "ranging_to_noise_psd, success_probability, code, expected_time, case_description",
    gen_pn_acq_time_test_params(),
)
def test_pn_acquisition_time(
    ranging_to_noise_psd: DecibelHertz,
    success_probability: Dimensionless,
    code: ranging.PnRangingCode,
    expected_time: Time,
    case_description: str,
):
    result = ranging.pn_acquisition_time(
        ranging_to_noise_psd, success_probability, code
    )

    assert_quantity_allclose(
        result,
        expected_time,
        rtol=5e-2,  # Test values are from plot extraction
        err_msg=f"Failed for case: {case_description}",
    )


@pytest.mark.parametrize(
    "range_clock_waveform, "
    "reference_clock_waveform, "
    "tracking_architecture, "
    "jitter_offset_from_dsn, "
    "case_description",
    [
        # The case from [2] Figures 12 and 13. No offset is required.
        (
            ranging.RangeClockWaveform.SINE,
            ranging.RangeClockWaveform.SINE,
            None,
            1.0,
            "sine-sine",
        ),
        (
            ranging.RangeClockWaveform.SINE,
            ranging.RangeClockWaveform.SQUARE,
            None,
            (4 * np.pi) / (8 * math.sqrt(2)),
            "sine-square",
        ),
        (
            ranging.RangeClockWaveform.SQUARE,
            ranging.RangeClockWaveform.SQUARE,
            ranging.TrackingArchitecture.OPEN_LOOP,
            (4 * np.pi) / (8 * math.sqrt(2)),
            "square-square open loop",
        ),
        (
            ranging.RangeClockWaveform.SQUARE,
            ranging.RangeClockWaveform.SQUARE,
            ranging.TrackingArchitecture.CLOSED_LOOP,
            (4 * np.pi) / 8,
            "square-square closed loop",
        ),
    ],
)
def test_range_est_variance_vs_dsn_plots(
    range_clock_waveform: ranging.RangeClockWaveform,
    reference_clock_waveform: ranging.RangeClockWaveform,
    tracking_architecture: ranging.TrackingArchitecture | None,
    jitter_offset_from_dsn: float,
    case_description: str,
):
    """Test _range_est_variance against plot-extracted reference data."""

    range_clock_rate = 1 * u.MHz
    integration_time = 1 * u.s
    loop_bandwidth = 1 / (2 * integration_time)

    # Data from [2] Figures 12 and 13, extracted with WebPlotDigitizer.
    # All of these are for a 1 MHz range clock.
    # Keys are T * P_R/N_0 in dB and values are range jitter in meters.
    jitter_plot_data = {
        ranging.PnRangingCode.DSN: {
            29.994: 0.55937,
            30.996: 0.49931,
            31.992: 0.44455,
            32.988: 0.39686,
            33.997: 0.35270,
            34.993: 0.31501,
            35.995: 0.28086,
            36.998: 0.25083,
            38.000: 0.22316,
        },
        ranging.PnRangingCode.CCSDS_T4B: {
            29.994: 0.56879,
            30.996: 0.50756,
            31.992: 0.45280,
            32.995: 0.40334,
            33.997: 0.35859,
            34.999: 0.31973,
            35.995: 0.28616,
            36.998: 0.25496,
            38.000: 0.22728,
        },
        ranging.PnRangingCode.CCSDS_T2B: {
            16.561: 4.0000,
            16.999: 3.8043,
            17.998: 3.3894,
            18.998: 3.0176,
            20.003: 2.6928,
            21.002: 2.3992,
            21.995: 2.1409,
            22.995: 1.9100,
            24.000: 1.7065,
        },
    }

    t_prcn0_db = []
    expected_jitter_m = []
    for code in jitter_plot_data:
        t_prn0_db_list, expected_jitter_m_list = zip(
            *jitter_plot_data[code].items(), strict=True
        )
        expected_jitter_m += expected_jitter_m_list

        # Convert from P_R/N_0 to P_{RC}/N_0 using the correlation coefficient for the
        # range clock component. This accounts for the fact that some of the ranging
        # power is allocated to the other components of the ranging signal.
        t_prcn0_db += list(
            np.array(t_prn0_db_list) + 20 * np.log10(ranging._CORR_COEFF_DSN[code][1])
        )
    t_prcn0_db = np.array(t_prcn0_db) * u.dB

    # We only have plots of the jitter expressions from [2], so to test other cases of
    # range clock waveform and reference clock waveform an appropriate offset is applied
    # to the DSN expected jitter values. This isn't ideal but it's the best we can do
    # with the limited reference data available.
    expected_jitter_m = jitter_offset_from_dsn * np.array(expected_jitter_m) * u.m

    range_variance = ranging._range_est_variance(
        range_clock_rate,
        ranging.RangeJitterParameters(
            loop_bandwidth,
            t_prcn0_db - integration_time.to(u.dB(u.s)),
            range_clock_waveform,
            reference_clock_waveform,
            tracking_architecture,
        ),
    )
    calculated_jitter = np.sqrt(range_variance)

    assert_quantity_allclose(
        calculated_jitter,
        expected_jitter_m,
        rtol=1e-2,  # Somewhat loose because ref data came from plot extraction
        err_msg=f"Failed for case: {case_description}",
    )


@pytest.mark.parametrize(
    "code, "
    "tracking_architecture, "
    "range_clock_waveform, "
    "reference_clock_waveform, "
    "expected_jitter_m",
    [
        # [4] Table 2-9
        (
            ranging.PnRangingCode.CCSDS_T4B,
            ranging.TrackingArchitecture.CLOSED_LOOP,
            ranging.RangeClockWaveform.SINE,
            ranging.RangeClockWaveform.SINE,
            0.78 * u.m,
        ),
        (
            ranging.PnRangingCode.CCSDS_T4B,
            ranging.TrackingArchitecture.CLOSED_LOOP,
            ranging.RangeClockWaveform.SINE,
            ranging.RangeClockWaveform.SQUARE,
            0.87 * u.m,
        ),
        (
            ranging.PnRangingCode.CCSDS_T4B,
            ranging.TrackingArchitecture.CLOSED_LOOP,
            ranging.RangeClockWaveform.SQUARE,
            ranging.RangeClockWaveform.SQUARE,
            1.22 * u.m,
        ),
        (
            ranging.PnRangingCode.CCSDS_T2B,
            ranging.TrackingArchitecture.CLOSED_LOOP,
            ranging.RangeClockWaveform.SINE,
            ranging.RangeClockWaveform.SINE,
            1.17 * u.m,
        ),
        (
            ranging.PnRangingCode.CCSDS_T2B,
            ranging.TrackingArchitecture.CLOSED_LOOP,
            ranging.RangeClockWaveform.SINE,
            ranging.RangeClockWaveform.SQUARE,
            1.29 * u.m,
        ),
        (
            ranging.PnRangingCode.CCSDS_T2B,
            ranging.TrackingArchitecture.CLOSED_LOOP,
            ranging.RangeClockWaveform.SQUARE,
            ranging.RangeClockWaveform.SQUARE,
            1.82 * u.m,
        ),
        # [4] Table 2-12 (some entries are redundant with Table 2-9; that's okay)
        (
            ranging.PnRangingCode.CCSDS_T4B,
            ranging.TrackingArchitecture.CLOSED_LOOP,
            ranging.RangeClockWaveform.SINE,
            ranging.RangeClockWaveform.SQUARE,
            0.87 * u.m,
        ),
        (
            ranging.PnRangingCode.CCSDS_T4B,
            ranging.TrackingArchitecture.OPEN_LOOP,
            ranging.RangeClockWaveform.SINE,
            ranging.RangeClockWaveform.SINE,
            0.78 * u.m,
        ),
        (
            ranging.PnRangingCode.CCSDS_T2B,
            ranging.TrackingArchitecture.CLOSED_LOOP,
            ranging.RangeClockWaveform.SINE,
            ranging.RangeClockWaveform.SQUARE,
            # Strangely this value is slightly different from the 1.29 m value given in
            # Table 2-9 for the same case. Unclear why; maybe different rounding?
            1.30 * u.m,
        ),
        (
            ranging.PnRangingCode.CCSDS_T2B,
            ranging.TrackingArchitecture.OPEN_LOOP,
            ranging.RangeClockWaveform.SINE,
            ranging.RangeClockWaveform.SINE,
            1.17 * u.m,
        ),
    ],
)
def test_range_est_variance_vs_ccsds_green_book(
    code: ranging.PnRangingCode,
    tracking_architecture: ranging.TrackingArchitecture,
    range_clock_waveform: ranging.RangeClockWaveform,
    reference_clock_waveform: ranging.RangeClockWaveform,
    expected_jitter_m: Distance,
):
    """Test _range_est_variance against [4] Tables 2-9 and 2-12."""
    # Parameters associated with Tables 2-9 and 2-12 in [4]
    loop_bandwidth = 1.0 * u.Hz
    chip_rate = 2.068 * u.MHz
    range_clock_rate = chip_rate / 2
    prc_n0_values = {
        ranging.PnRangingCode.CCSDS_T4B: 29.45 * u.dBHz,
        ranging.PnRangingCode.CCSDS_T2B: 25.95 * u.dBHz,
    }

    range_variance = ranging._range_est_variance(
        range_clock_rate,
        ranging.RangeJitterParameters(
            loop_bandwidth,
            prc_n0_values[code],
            range_clock_waveform,
            reference_clock_waveform,
            tracking_architecture,
        ),
    )
    calculated_jitter = np.sqrt(range_variance)

    assert_quantity_allclose(
        calculated_jitter,
        expected_jitter_m,
        atol=1e-2 * u.m,  # Green book table values are rounded to 2 decimal places
    )


def test_range_jitter_coefficient_error_cases():
    with pytest.raises(ValueError):
        ranging._range_jitter_coefficient(
            ranging.RangeClockWaveform.SQUARE,
            ranging.RangeClockWaveform.SQUARE,
            None,
        )
    with pytest.raises(ValueError):
        ranging._range_jitter_coefficient(
            ranging.RangeClockWaveform.SQUARE,
            ranging.RangeClockWaveform.SINE,
            ranging.TrackingArchitecture.CLOSED_LOOP,
        )
