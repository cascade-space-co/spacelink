import enum

import astropy.units as u
import numpy as np
import pydantic
import pydantic.dataclasses
import scipy.interpolate
import typing

from spacelink.phy.mode import LinkMode
from spacelink.core.units import Dimensionless, Decibels, enforce_units


class ErrorMetric(str, enum.Enum):
    BER = "bit error rate"
    WER = "codeword error rate"


class ModePerformance(pydantic.BaseModel):
    r"""
    Performance characteristics for specific link modes.

    This class provides methods to convert between Eb/N0 and error rates for given
    modulation and coding schemes.

    Parameters
    ----------
    modes : list[LinkMode]
        The link mode configurations.
    decoder_profile : DecoderProfile
        Configuration for the decoder stages.
    metric : ErrorMetric
        Type of error metric (bit error rate, codeword error rate, etc.).
    points : list[tuple[float, float]]
        List of (Eb/N0 [dB], error rate) data points for interpolation.
    ref : str, optional
        Reference or source of the performance data (default: "").
    """

    modes: list[LinkMode]
    metric: ErrorMetric
    points: list[tuple[float, float]]  # (Eb/N0 [dB], error rate)
    ref: str = ""

    def __init__(self, **data):
        super().__init__(**data)
        self._create_interpolators()

    def _create_interpolators(self) -> None:
        """Create interpolator objects for efficient reuse."""
        points = np.array(self.points)
        ebno_values = points[:, 0]
        error_rate_values = points[:, 1]

        # Create interpolator for Eb/N0 -> error rate
        self._ebno_to_error_interpolator = scipy.interpolate.PchipInterpolator(
            ebno_values,
            np.log10(error_rate_values),
            extrapolate=False,
        )

        # Create interpolator for error rate -> Eb/N0
        # Need to sort by error rate (ascending) so log values are increasing
        sorted_indices = np.argsort(error_rate_values)
        sorted_error_rates = error_rate_values[sorted_indices]
        sorted_ebno_values = ebno_values[sorted_indices]

        self._error_to_ebno_interpolator = scipy.interpolate.PchipInterpolator(
            np.log10(sorted_error_rates),
            sorted_ebno_values,
            extrapolate=False,
        )

    @enforce_units
    def ebno_to_error_rate(self, ebno: Decibels) -> Dimensionless:
        r"""
        Find the error rate corresponding to the given Eb/N0.

        Parameters
        ----------
        ebno : Decibels
            Energy per bit to noise power spectral density ratio :math:`E_b/N_0`.

        Returns
        -------
        Dimensionless
            Error rate or NaN if the Eb/N0 is outside the range of available performance
            data. Same shape as `ebno`.
        """
        return 10.0 ** self._ebno_to_error_interpolator(ebno.value) * u.dimensionless

    @enforce_units
    def error_rate_to_ebno(self, error_rate: Dimensionless) -> Decibels:
        r"""
        Find Eb/N0 required to achieve the target error rate.

        Parameters
        ----------
        error_rate : Dimensionless
            Target error rate.

        Returns
        -------
        Decibels
            Required Eb/N0 in decibels to achieve the target error rate or NaN if the
            error rate is outside the range of available performance data. Same shape as
            `error_rate`.
        """
        return self._error_to_ebno_interpolator(np.log10(error_rate.value)) * u.dB

    @enforce_units
    def coding_gain(self, uncoded: typing.Self, error_rate: Dimensionless) -> Decibels:
        r"""
        Calculate the coding gain relative to an uncoded reference.

        The coding gain is the difference in required Eb/N0 between the uncoded and
        coded systems at the same error rate.

        Parameters
        ----------
        uncoded : ModePerformance
            Performance model for the uncoded reference system. Must use the same error
            metric as this object.
        error_rate : Dimensionless
            Error rate at which to evaluate the coding gain.

        Returns
        -------
        Decibels
            Coding gain in decibels or NaN if the error rate is outside the range of
            the available performance data. Same shape as `error_rate`.

        Raises
        ------
        ValueError
            If the uncoded model has a different error metric.
        """
        if uncoded.metric != self.metric:
            raise ValueError(f"Uncoded metric {uncoded.metric} ≠ {self.metric}.")

        uncoded_ebno = uncoded.error_rate_to_ebno(error_rate)
        coded_ebno = self.error_rate_to_ebno(error_rate)
        return (uncoded_ebno - coded_ebno) * u.dB
