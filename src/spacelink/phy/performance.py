import enum
import typing

import astropy.units as u
import numpy as np
import pydantic
import scipy.interpolate

from spacelink.core.units import Decibels, Dimensionless, enforce_units
from spacelink.phy.mode import LinkMode

ErrorCurvePoint = tuple[float, float]  # (Eb/N0 [dB], error rate)


class ErrorMetric(str, enum.Enum):
    BER = "bit error rate"
    WER = "codeword error rate"
    FER = "frame error rate"


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
    points : list[ErrorCurvePoint]
        List of error rate curve data points for interpolation.
    ref : str, optional
        Reference or source of the performance data (default: "").
    """

    modes: list[LinkMode]
    metric: ErrorMetric
    points: list[ErrorCurvePoint]
    ref: str = ""

    @pydantic.field_validator("points")
    @classmethod
    def validate_minimum_points(cls, v):
        if len(v) < 2:
            raise ValueError(
                "ModePerformance requires at least two data points for interpolation"
            )
        return v

    @pydantic.field_validator("points")
    @classmethod
    def validate_points_sorted(cls, v):
        ebn0_values = np.array([point[0] for point in v])
        if not np.all(np.diff(ebn0_values) > 0):
            raise ValueError("Points must be sorted in strictly increasing Eb/N0 order")
        return v

    @pydantic.field_validator("points")
    @classmethod
    def validate_error_values_decreasing(cls, v):
        error_values = np.array([point[1] for point in v])
        if not np.all(np.diff(error_values) < 0):
            raise ValueError(
                "Error values must be strictly decreasing with increasing Eb/N0"
            )
        return v

    def __init__(self, **data):
        super().__init__(**data)
        self._create_interpolators()

    def _create_interpolators(self) -> None:
        """Create interpolator objects for efficient reuse."""
        points = np.array(self.points)
        ebn0_values = points[:, 0]
        error_rate_values = points[:, 1]

        # Create interpolator for Eb/N0 -> error rate
        # Points are guaranteed to be sorted by Eb/N0 due to validation
        self._ebn0_to_error_interpolator = scipy.interpolate.PchipInterpolator(
            ebn0_values,
            np.log10(error_rate_values),
            extrapolate=False,
        )

        # Create interpolator for error rate -> Eb/N0
        # Need to sort by error rate (ascending) so log values are increasing
        sorted_indices = np.argsort(error_rate_values)
        sorted_error_rates = error_rate_values[sorted_indices]
        sorted_ebn0_values = ebn0_values[sorted_indices]

        self._error_to_ebn0_interpolator = scipy.interpolate.PchipInterpolator(
            np.log10(sorted_error_rates),
            sorted_ebn0_values,
            extrapolate=False,
        )

    @enforce_units
    def ebn0_to_error_rate(self, ebn0: Decibels) -> Dimensionless:
        r"""
        Find the error rate corresponding to the given Eb/N0.

        Parameters
        ----------
        ebn0 : Decibels
            Energy per bit to noise power spectral density ratio :math:`E_b/N_0`.

        Returns
        -------
        Dimensionless
            Error rate or NaN if the Eb/N0 is outside the range of available performance
            data. Same shape as ``ebn0``.
        """
        return 10.0 ** self._ebn0_to_error_interpolator(ebn0.value) * u.dimensionless

    @enforce_units
    def error_rate_to_ebn0(self, error_rate: Dimensionless) -> Decibels:
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
            ``error_rate``.
        """
        return self._error_to_ebn0_interpolator(np.log10(error_rate.value)) * u.dB

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
            the available performance data. Same shape as ``error_rate``.

        Raises
        ------
        ValueError
            If the uncoded model has a different error metric.
        """
        if uncoded.metric != self.metric:
            raise ValueError(f"Uncoded metric {uncoded.metric} ≠ {self.metric}.")

        uncoded_ebn0 = uncoded.error_rate_to_ebn0(error_rate)
        coded_ebn0 = self.error_rate_to_ebn0(error_rate)
        return uncoded_ebn0 - coded_ebn0
