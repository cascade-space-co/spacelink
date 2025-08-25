"""
Calculations related to two-way sequential or pseudo-noise (PN) radiometric ranging.

This module provides functions for calculating range ambiguity and power allocations
between residual carrier and modulated components.

References
----------
`[1]`_ 810-005 203, Rev. D "Sequential Ranging"

`[2]`_ 810-005 214, Rev. C "Pseudo-Noise and Regenerative Ranging"

`[3]`_ CCSDS 414.1-B-3 "Pseudo-Noise (PN) Ranging Systems Recommended Standard"

`[4]`_ CCSDS 414.0-G-2 "Pseudo-Noise (PN) Ranging Systems Informational Report"

.. _[1]: https://deepspace.jpl.nasa.gov/dsndocs/810-005/203/203D.pdf
.. _[2]: https://deepspace.jpl.nasa.gov/dsndocs/810-005/214/214C.pdf
.. _[3]: https://ccsds.org/wp-content/uploads/gravity_forms/\
5-448e85c647331d9cbaf66c096458bdd5/2025/01//414x1b3e1.pdf
.. _[4]: https://ccsds.org/wp-content/uploads/gravity_forms/\
5-448e85c647331d9cbaf66c096458bdd5/2025/01//414x0g2.pdf
"""

import enum
import math

import astropy.constants as const
import astropy.units as u
import numpy as np
import scipy.integrate
import scipy.optimize
import scipy.special

from .units import (
    Angle,
    DecibelHertz,
    Decibels,
    Dimensionless,
    Distance,
    Frequency,
    Time,
    enforce_units,
)


class PnRangingCode(enum.Enum):
    """The type of PN ranging code used."""

    DSN = enum.auto()
    CCSDS_T4B = enum.auto()
    CCSDS_T2B = enum.auto()


class DataModulation(enum.Enum):
    """The type of data modulation used alongside ranging."""

    BIPOLAR = enum.auto()
    SINE_SUBCARRIER = enum.auto()


CODE_LENGTH = 1_009_470
"""int: The length of the full PN ranging sequence.

The DSN and CCSDS PN ranging codes all have the same length.

References
----------
`[2]`_ Equation (9).

`[3]`_ Sections 3.2.2 and 3.2.3.
"""


COMPONENT_LENGTHS = {
    1: 2,
    2: 7,
    3: 11,
    4: 15,
    5: 19,
    6: 23,
}
"""dict[int, int]: The lengths of the six components of the PN ranging codes.

The DSN and CCSDS PN ranging codes all have the same component lengths.

The key is the component index (1-6).
The value is the length of the component in chips.

References
----------
`[2]`_ Table 2.

`[3]`_ Sections 3.2.2 and 3.2.3.
"""


@enforce_units
def pn_sequence_range_ambiguity(ranging_clock_rate: Frequency) -> Distance:
    r"""
    Compute the range ambiguity of the standard PN ranging sequences.

    Parameters
    ----------
    ranging_clock_rate : Frequency
        Rate of the ranging clock :math:`f_{RC}`. This is half the chip rate.

    Returns
    -------
    Distance
        The range ambiguity distance.

    References
    ----------
    `[2]`_ Equation (11).

    `[4]`_ p. 2-2.
    """
    return (CODE_LENGTH * const.c / (4 * ranging_clock_rate)).decompose()


@enforce_units
def chip_snr(ranging_clock_rate: Frequency, prn0: DecibelHertz) -> Decibels:
    r"""
    Compute the chip SNR :math:`2E_C/N_0` in decibels.

    Parameters
    ----------
    ranging_clock_rate : Frequency
        Rate of the ranging clock :math:`f_{RC}`. This is half the chip rate.
    prn0 : DecibelHertz
        The ranging signal-to-noise spectral density ratio :math:`P_R/N_0`.

    Returns
    -------
    Decibels
        The chip SNR :math:`2E_C/N_0`.

    References
    ----------
    `[4]`_ p. 2-3.
    """
    return prn0 - ranging_clock_rate.to(u.dB(u.Hz))


@enforce_units
def _suppression_factor(mod_idx: Angle, modulation: DataModulation) -> Dimensionless:
    r"""
    Compute the suppression factor :math:`S_{cmd}(\phi_{cmd})`.

    This is used in the expressions for carrier and ranging power fractions.

    Parameters
    ----------
    mod_idx : Angle
        The RMS phase deviation by command signal :math:`\phi_{cmd}`.
    modulation : DataModulation
        The data modulation type.

    Returns
    -------
    Dimensionless
        The suppression factor :math:`S_{cmd}(\phi_{cmd})`.

    References
    ----------
    `[1]`_ Equation (15).

    `[2]`_ Equation (24).
    """
    mod_idx_rad = mod_idx.value
    if modulation == DataModulation.BIPOLAR:
        suppression_factor = np.cos(mod_idx_rad) ** 2
    elif modulation == DataModulation.SINE_SUBCARRIER:
        suppression_factor = scipy.special.j0(math.sqrt(2) * mod_idx_rad) ** 2
    else:
        raise ValueError(f"Invalid data modulation type: {modulation}")
    return suppression_factor * u.dimensionless_unscaled


@enforce_units
def _modulation_factor(mod_idx: Angle, modulation: DataModulation) -> Dimensionless:
    r"""
    Compute the modulation factor :math:`M_{cmd}(\phi_{cmd})`.

    This is used in the expression for data power fraction.

    Parameters
    ----------
    mod_idx : Angle
        The RMS phase deviation by command signal :math:`\phi_{cmd}`.
    modulation : DataModulation
        The data modulation type.

    Returns
    -------
    Dimensionless
        The modulation factor :math:`M_{cmd}(\phi_{cmd})`.

    References
    ----------
    `[1]`_ Equation (16).

    `[2]`_ Equation (25).
    """
    mod_idx_rad = mod_idx.value
    if modulation == DataModulation.BIPOLAR:
        mod_factor = np.sin(mod_idx_rad) ** 2
    elif modulation == DataModulation.SINE_SUBCARRIER:
        mod_factor = 2 * scipy.special.j1(math.sqrt(2) * mod_idx_rad) ** 2
    else:
        raise ValueError(f"Invalid data modulation type: {modulation}")
    return mod_factor * u.dimensionless_unscaled


@enforce_units
def carrier_to_total_power(
    mod_idx_ranging: Angle,
    mod_idx_data: Angle,
    modulation: DataModulation,
) -> Dimensionless:
    r"""
    Ratio of residual carrier power to total power :math:`P_{C}/P_{T}`.

    This applies under the following conditions:

    * The ranging clock (chip pulse shape in the case of PN ranging) is a sinewave.
    * Uplink or regenerative downlink.

    This does not apply to the downlink case when a transparent (non-regenerative)
    transponder is used.

    Parameters
    ----------
    mod_idx_ranging : Angle
        The RMS phase deviation by ranging signal; :math:`\phi_{r}` for uplink or
        :math:`\theta_{rs}` for downlink.
    mod_idx_data : Angle
        The RMS phase deviation by data signal; :math:`\phi_{cmd}` for uplink or
        :math:`\theta_{tlm}` for downlink.
    modulation : DataModulation
        The data modulation type.

    Returns
    -------
    Dimensionless
        The ratio of residual carrier power to total power :math:`P_{C}/P_{T}`.

    References
    ----------
    `[1]`_ Equation (10) for sequential ranging uplink.

    `[2]`_ Equation (19) for PN ranging uplink, (50) for regenerative downlink.
    """
    return (
        scipy.special.j0(math.sqrt(2) * mod_idx_ranging.value) ** 2
        * _suppression_factor(mod_idx_data, modulation)
    ) * u.dimensionless_unscaled


@enforce_units
def ranging_to_total_power(
    mod_idx_ranging: Angle,
    mod_idx_data: Angle,
    modulation: DataModulation,
) -> Dimensionless:
    r"""
    Ratio of usable ranging power to total power :math:`P_{R}/P_{T}`.

    This applies under the following conditions:

    * The ranging clock (chip pulse shape in the case of PN ranging) is a sinewave.
    * Uplink or regenerative downlink.

    This does not apply to the downlink case when a transparent (non-regenerative)
    transponder is used.

    Parameters
    ----------
    mod_idx_ranging : Angle
        The RMS phase deviation by ranging signal; :math:`\phi_{r}` for uplink or
        :math:`\theta_{rs}` for downlink.
    mod_idx_data : Angle
        The RMS phase deviation by data signal; :math:`\phi_{cmd}` for uplink or
        :math:`\theta_{tlm}` for downlink.
    modulation : DataModulation
        The data modulation type.

    Returns
    -------
    Dimensionless
        The ratio of usable ranging power to total power :math:`P_{R}/P_{T}`.

    References
    ----------
    `[1]`_ Equation (11) for sequential ranging uplink.

    `[2]`_ Equation (20) for PN ranging uplink, (51) for regenerative downlink.
    """
    return (
        2
        * scipy.special.j1(math.sqrt(2) * mod_idx_ranging.value) ** 2
        * _suppression_factor(mod_idx_data, modulation)
    ) * u.dimensionless_unscaled


@enforce_units
def data_to_total_power(
    mod_idx_ranging: Angle,
    mod_idx_data: Angle,
    modulation: DataModulation,
) -> Dimensionless:
    r"""
    Ratio of usable data power to total power :math:`P_{D}/P_{T}`.

    This applies under the following conditions:

    * The ranging clock (chip pulse shape in the case of PN ranging) is a sinewave.
    * Uplink or regenerative downlink.

    This does not apply to the downlink case when a transparent (non-regenerative)
    transponder is used.

    Parameters
    ----------
    mod_idx_ranging : Angle
        The RMS phase deviation by ranging signal; :math:`\phi_{r}` for uplink or
        :math:`\theta_{rs}` for downlink.
    mod_idx_data : Angle
        The RMS phase deviation by data signal; :math:`\phi_{cmd}` for uplink or
        :math:`\theta_{tlm}` for downlink.
    modulation : DataModulation
        The data modulation type.

    Returns
    -------
    Dimensionless
        The ratio of usable data power to total power :math:`P_{D}/P_{T}`.

    References
    ----------
    `[1]`_ Equation (12) for sequential ranging uplink.

    `[2]`_ Equation (21) for PN ranging uplink, (52) for regenerative downlink.
    """
    return (
        scipy.special.j0(math.sqrt(2) * mod_idx_ranging.value) ** 2
        * _modulation_factor(mod_idx_data, modulation)
    ) * u.dimensionless_unscaled


# Values of the cross-correlation factors R_n given in Tables 3, 4, and 5 of [2].
_CORR_COEFF_DSN = {
    # [2] Table 3
    PnRangingCode.DSN: {
        1: 0.9544,
        2: 0.0456,
        3: 0.0456,
        4: 0.0456,
        5: 0.0456,
        6: 0.0456,
    },
    # [2] Table 4
    PnRangingCode.CCSDS_T4B: {
        1: 0.9387,
        2: 0.0613,
        3: 0.0613,
        4: 0.0613,
        5: 0.0613,
        6: 0.0613,
    },
    # [2] Table 5
    PnRangingCode.CCSDS_T2B: {
        1: 0.6274,
        2: 0.2447,
        3: 0.2481,
        4: 0.2490,
        5: 0.2492,
        6: 0.2496,
    },
}


@enforce_units
def pn_component_acquisition_probability(
    ranging_to_noise_psd: Frequency,
    integration_time: Time,
    code: PnRangingCode,
    component: int,
) -> Dimensionless:
    r"""
    Compute the acquisition probability for one component of the PN ranging code.

    Successful acquisition means that the correct phase of the component code is
    identified by the receiver. The number of possible phases is equal to the component
    length.

    This calculation assumes that the received signal is correlated against all
    possible cyclic shifts of the component code in parallel and that the shift with the
    maximum correlation output is selected. See `[4]`_ Section 2.6.2 for details.

    This function uses equation (91) from `[2]`_. `[4]`_ has an equivalent equation in
    Section 2.6.3.2, but it uses slightly different values for the cross-correlation
    coefficients which leads to slightly different acquisition probabilities.

    Parameters
    ----------
    ranging_to_noise_psd : Frequency
        The ranging-to-noise power spectral density :math:`P_R/N_0`.
    integration_time : Time
        The integration time :math:`T`.
    code : PnRangingCode
        The PN ranging code type.
    component : int
        The component index in [1, 6].

    Returns
    -------
    Dimensionless
        The acquisition probability for the given component.

    References
    ----------
    `[2]`_ Equation (91).
    """
    comp_len = COMPONENT_LENGTHS[component]
    corr_coeff = _CORR_COEFF_DSN[code][component]
    corr_term = corr_coeff * math.sqrt(integration_time * ranging_to_noise_psd)
    integral = scipy.integrate.quad(
        lambda x: (
            math.exp(-(x**2))
            * ((1 + scipy.special.erf(x + corr_term)) / 2) ** (comp_len - 1)
        ),
        -np.inf,
        np.inf,
    )[0]
    return integral / math.sqrt(np.pi) * u.dimensionless


@enforce_units
def pn_acquisition_probability(
    ranging_to_noise_psd: Frequency, integration_time: Time, code: PnRangingCode
) -> Dimensionless:
    r"""
    Compute the overall acquisition probability for the PN ranging code.

    This is the product of the acquisition probabilities for all components.

    This calculation assumes that correlations for all component codes are performed in
    parallel. See `[4]`_ Section 2.6.2 for details.

    Parameters
    ----------
    ranging_to_noise_psd : Frequency
        The ranging-to-noise power spectral density :math:`P_R/N_0`.
    integration_time : Time
        The integration time :math:`T`.
    code : PnRangingCode
        The PN ranging code type.

    Returns
    -------
    Dimensionless
        The overall acquisition probability for the PN ranging code.

    References
    ----------
    `[2]`_ Equation (90).
    """
    return (
        float(
            np.prod(
                [
                    pn_component_acquisition_probability(
                        ranging_to_noise_psd, integration_time, code, component
                    )
                    for component in range(2, 7)
                ]
            )
        )
        * u.dimensionless
    )


@enforce_units
def pn_acquisition_time(
    ranging_to_noise_psd: Frequency,
    success_probability: Dimensionless,
    code: PnRangingCode,
) -> Time:
    r"""
    Compute the acquisition time for the PN ranging code.

    Successful acquisition means that the correct phase of the full PN ranging sequence
    is identified by the receiver. For the same receiver architecture, higher
    probability of successs requires either higher :math:`P_R/N_0` or more time.

    This calculation assumes that correlations for all component codes are performed in
    parallel, which is typical for ground station receivers. `[3]`_ and `[4]`_ refer to
    this as "station acquisition," though it could also apply to a regenerative
    transponder that uses the same parallel correlation architecture. The See `[4]`_
    Section 2.6.2 for details.

    Keep in mind that this only accounts for the acquisition time of a single PN ranging
    receiver. If a regenerative transponder is used, the ground station receiver's
    acquisition processing can't begin until the on-board transponder has completed
    its acquisition processing.

    Parameters
    ----------
    ranging_to_noise_psd : Frequency
        The ranging-to-noise power spectral density :math:`P_R/N_0`.
    success_probability : Dimensionless
        The desired probability of successful acquisition.
    code : PnRangingCode
        The PN ranging code type.

    Returns
    -------
    Time
        The minimum integration time required to achieve the desired probability of
        successful acquisition.

    References
    ----------
    `[2]`_ Inverse of Equation (90).
    """
    if not 0 < success_probability < 1:
        raise ValueError("Success probability must be between 0 and 1")

    # The following are upper bounds on the expected acquisition time required for the
    # root finding algorithm. These were found by fitting curves to the acquisition
    # probability equations and then adding some margin.
    if code == PnRangingCode.DSN:
        upper_bound = (
            2000 + 2166 * -math.log10(1 - success_probability)
        ) / ranging_to_noise_psd
    elif code == PnRangingCode.CCSDS_T4B:
        upper_bound = (
            1100 + 1198 * -math.log10(1 - success_probability)
        ) / ranging_to_noise_psd
    elif code == PnRangingCode.CCSDS_T2B:
        upper_bound = (
            100 + 73 * -math.log10(1 - success_probability)
        ) / ranging_to_noise_psd
    else:
        raise ValueError(f"Invalid PN ranging code: {code}")

    # There's no closed-form solution to Equation (90) or (91) so we need to use a
    # root-finding algorithm.
    solution = scipy.optimize.root_scalar(
        lambda int_time: (
            pn_acquisition_probability(ranging_to_noise_psd, int_time * u.s, code)
            - success_probability
        ),
        bracket=(0.0, upper_bound.value),
    )

    return solution.root * u.s
