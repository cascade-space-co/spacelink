import enum

import astropy.units as u
import numpy as np
import pydantic
import scipy.interpolate
import typing

from spacelink.phy.mode import LinkMode
from spacelink.core.units import Dimensionless, Decibels, enforce_units


class ErrorMetric(str, enum.Enum):
    BER = "bit error rate"
    WER = "codeword error rate"


class DecoderStage(pydantic.BaseModel):
    r"""
    Configuration for a single decoder stage. This can be used to represent any kind of
    decoding stage, but a demodulation stage or error correction decoding stage are the
    common cases.
    
    It's most relevant to use this class when the performance of the modulation and 
    coding depends on some property of a stage decoding stage, such as the number of 
    decoder iterations or the number of soft bits in a soft-decision decoder. In such
    cases there may be different performance curves for different property values.

    Parameters
    ----------
    type : str
        Type of decoder stage (e.g., 'rs', 'conv', 'ldpc', 'modulation').
    algorithm : str
        Name of the decoding algorithm.
    parameters : dict
        Additional parameters for the decoder stage. Examples include hard versus soft
        decision decoding, number of soft bits, number of decoder iterations, etc.
        Properties of the modulation and coding that affect compatibility between 
        transmitter and receiver should not be included.
    """
    type: str  # e.g., 'rs', 'conv', 'ldpc', 'modulation'
    algorithm: str
    parameters: dict[str, typing.Any] = {}


class DecoderProfile(pydantic.BaseModel):
    r"""
    Configuration profile for a full decoder pipeline.

    Parameters
    ----------
    stages : list[DecoderStage]
        List of decoder stages in processing order.
    """
    stages: list[DecoderStage]


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
    decoder_profile: DecoderProfile
    metric: ErrorMetric
    points: list[tuple[float, float]]  # (Eb/N0 [dB], error rate)
    ref: str = ""

    @pydantic.field_validator('modes')
    @classmethod
    def validate_modes_not_empty(cls, v):
        """Validate that the modes list contains at least one item."""
        if not v:
            raise ValueError("modes list must contain at least one item")
        return v

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
    def coding_gain(
        self, uncoded_model: typing.Self, error_rate: Dimensionless
    ) -> Decibels:
        r"""
        Calculate the coding gain relative to an uncoded reference.

        The coding gain is the difference in required Eb/N0 between the uncoded and 
        coded systems at the same error rate.

        Parameters
        ----------
        uncoded_model : ModePerformance
            Performance model for the uncoded reference system.
        error_rate : Dimensionless
            Error rate at which to evaluate the coding gain.

        Returns
        -------
        Decibels
            Coding gain in decibels or NaN if the error rate is outside the range of
            the available performance data. Same shape as `error_rate`.
        """
        uncoded_ebno = uncoded_model.error_rate_to_ebno(error_rate)
        coded_ebno = self.error_rate_to_ebno(error_rate)
        return (uncoded_ebno - coded_ebno) * u.dB
