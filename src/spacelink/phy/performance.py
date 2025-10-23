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
    PER = "packet error rate"


class ModePerformanceCurve(pydantic.BaseModel):
    r"""
    Performance characteristics for specific link modes with interpolatable curves.

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
        return self._error_to_ebn0_interpolator(np.log10(error_rate.value)) * u.dB(1)

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


class ModePerformanceThreshold(pydantic.BaseModel):
    r"""
    Operating point threshold performance for specific link modes.

    This class represents performance data with a single operating point threshold,
    typically used for standards like DVB-S2 where only quasi-error-free (QEF)
    operating points are published.

    Parameters
    ----------
    modes : list[LinkMode]
        The link mode configurations.
    metric : ErrorMetric
        Type of error metric (bit error rate, codeword error rate, etc.).
    ebn0 : float
        Eb/N0 threshold in dB for quasi-error-free operation.
    error_rate : float
        Error rate at the threshold point (dimensionless).
    ref : str, optional
        Reference or source of the performance data (default: "").
    """

    modes: list[LinkMode]
    metric: ErrorMetric
    ebn0: float
    error_rate: float
    ref: str = ""

    @property
    def threshold_ebn0(self) -> Decibels:
        """
        Get the Eb/N0 threshold with proper units.

        Returns
        -------
        Decibels
            Eb/N0 threshold value.
        """
        return self.ebn0 * u.dB(1)

    @property
    def threshold_error_rate(self) -> Dimensionless:
        """
        Get the error rate threshold with proper units.

        Returns
        -------
        Dimensionless
            Error rate threshold value.
        """
        return self.error_rate * u.dimensionless

    @enforce_units
    def meets_threshold(self, ebn0: Decibels) -> bool | np.ndarray:
        """
        Check if the given Eb/N0 meets or exceeds the quasi-error-free threshold.

        Parameters
        ----------
        ebn0 : Decibels
            Energy per bit to noise power spectral density ratio :math:`E_b/N_0`.

        Returns
        -------
        bool or np.ndarray
            True where Eb/N0 meets or exceeds the threshold. Scalar if input is scalar,
            array if input is array.
        """
        result = ebn0.value >= self.ebn0

        # Return scalar if input was scalar
        if np.isscalar(ebn0.value):
            return bool(result)
        return result

    @enforce_units
    def margin_to_threshold(self, ebn0: Decibels) -> Decibels:
        """
        Calculate the margin (positive) or shortfall (negative) relative to threshold.

        Parameters
        ----------
        ebn0 : Decibels
            Energy per bit to noise power spectral density ratio :math:`E_b/N_0`.

        Returns
        -------
        Decibels
            Margin in dB. Positive values indicate the link exceeds the threshold,
            negative values indicate it falls short. Same shape as ``ebn0``.
        """
        return (ebn0.value - self.ebn0) * u.dB(1)

    def coding_gain(self, uncoded: typing.Self) -> Decibels:
        r"""
        Calculate the coding gain relative to an uncoded threshold.

        The coding gain is the difference in required Eb/N0 between the uncoded and
        coded systems at the threshold error rate.

        Parameters
        ----------
        uncoded : ModePerformanceThreshold
            Threshold specification for the uncoded reference system. Must use the same
            error metric and have the same error rate threshold as this object.

        Returns
        -------
        Decibels
            Coding gain in decibels.

        Raises
        ------
        ValueError
            If the uncoded threshold has a different error metric or error rate.
        """
        if uncoded.metric != self.metric:
            raise ValueError(f"Uncoded metric {uncoded.metric} ≠ {self.metric}.")

        if not np.isclose(uncoded.error_rate, self.error_rate, rtol=1e-9, atol=1e-15):
            raise ValueError(
                f"Error rates must match for threshold comparison. "
                f"Uncoded error rate {uncoded.error_rate} ≠ {self.error_rate}."
            )

        return (uncoded.ebn0 - self.ebn0) * u.dB(1)


# Backwards compatibility alias
ModePerformance = ModePerformanceCurve
