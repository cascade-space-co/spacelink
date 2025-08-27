r"""
Polarization Loss
-----------------

The polarization loss between two antennas with given axial ratios is
calculated using the standard formula for polarization mismatch:

.. math::
   \text{PLF} = \frac{1}{2} +\
   \frac{1}{2} \frac{4 \gamma_1 \gamma_2 -\
   (1-\gamma_1^2)(1-\gamma_2^2)}{(1+\gamma_1^2)(1+\gamma_2^2)}

where:

* :math:`\gamma_1` and :math:`\gamma_2` are the voltage axial ratios (linear, not dB)
* PLF is the polarization loss factor (linear)

The polarization loss in dB is then:

.. math::
   L_{\text{pol}} = -10 \log_{10}(\text{PLF})

For circular polarization, the axial ratio is 0 dB, and for linear polarization,
it is >40 dB.

Dish Gain
---------

The gain of a parabolic dish antenna is given by:

.. math::
   G = \eta \left(\frac{\pi D}{\lambda}\right)^2

where:

* :math:`\eta` is the efficiency factor (typically 0.55 to 0.70)
* :math:`D` is the diameter of the dish
* :math:`\lambda` is the wavelength

Spherical Coordinate System
---------------------------

This module uses the standard spherical coordinate system with the following
conventions:

* :math:`\theta` is the polar angle measured from the +z axis with range [0, π] radians.
* :math:`\phi` is the azimuthal angle measured from the +x axis in the xy-plane with
  range [0, 2π) or [-π, π) radians.
"""

import enum
import typing

import astropy.units as u
import numpy as np
import scipy.integrate
import scipy.interpolate

from .units import (
    Angle,
    DecibelPerKelvin,
    Decibels,
    Dimensionless,
    Frequency,
    Length,
    SolidAngle,
    Temperature,
    enforce_units,
    safe_negate,
    to_dB,
    to_linear,
    wavelength,
)

__all__ = [
    "polarization_loss",
    "dish_gain",
    "Handedness",
    "Polarization",
    "RadiationPattern",
]


@enforce_units
def polarization_loss(ar1: Decibels, ar2: Decibels) -> Decibels:
    r"""
    Calculate the polarization loss in dB between two antennas with given axial ratios.

    Parameters
    ----------
    ar1 : Decibels
        First antenna axial ratio in dB (amplitude ratio)
    ar2 : Decibels
        Second antenna axial ratio in dB (amplitude ratio)

    Returns
    -------
    Decibels
        Polarization loss in dB (positive value)
    """
    # Polarization mismatch angle is omitted (assumed to be 90 degrees)
    # https://www.microwaves101.com/encyclopedias/polarization-mismatch-between-antennas
    gamma1 = to_linear(ar1, factor=20)
    gamma2 = to_linear(ar2, factor=20)

    numerator = 4 * gamma1 * gamma2 - (1 - gamma1**2) * (1 - gamma2**2)
    denominator = (1 + gamma1**2) * (1 + gamma2**2)

    # Calculate polarization loss factor
    plf = 0.5 + 0.5 * (numerator / denominator)
    return safe_negate(to_dB(plf))


@enforce_units
def dish_gain(
    diameter: Length, frequency: Frequency, efficiency: Dimensionless
) -> Decibels:
    r"""
    Calculate the gain in dB of a parabolic dish antenna.

    Parameters
    ----------
    diameter : Length
        Dish diameter
    frequency : Frequency
        Frequency
    efficiency : Dimensionless
        Antenna efficiency (dimensionless)

    Returns
    -------
    Decibels
        Gain in decibels (dB)

    Raises
    ------
    ValueError
        If frequency is not positive
    """
    if frequency <= 0 * u.Hz:
        raise ValueError("Frequency must be positive")

    wl = wavelength(frequency)
    gain_linear = efficiency * (np.pi * diameter.to(u.m) / wl) ** 2
    return to_dB(gain_linear)


class Handedness(enum.Enum):
    """Handedness of the polarization ellipse.

    The handedness is the direction of rotation of the E-field. The thumb points in the
    direction of propagation, and the fingers curl in the direction of the E-field
    rotation. When looking in the direction of propagation, the E-field rotates counter-
    clockwise for left-hand polarization and clockwise for right-hand polarization.
    """

    LEFT = enum.auto()
    RIGHT = enum.auto()


class Polarization:
    """Represents a polarization state."""

    @enforce_units
    def __init__(
        self,
        tilt_angle: Angle,
        axial_ratio: Dimensionless,
        handedness: Handedness,
    ):
        r"""
        Create a polarization state.

        Parameters
        ----------
        tilt_angle: Angle
            Tilt angle of the major axis of the polarization ellipse, measured in the
            local tangent plane, relative to :math:`\hat{\theta}`.
        axial_ratio: Dimensionless
            Ratio of the major to minor axis of the polarization ellipse.
        handedness: Handedness
            The direction of rotation of the E-field when looking in the direction of
            propagation.
        """

        if axial_ratio < 1:
            raise ValueError("Axial ratio must be ≥ 1 (≥ 0 dB)")

        self.tilt_angle = tilt_angle
        self.axial_ratio = axial_ratio
        self.handedness = handedness

        sign = -1 if handedness == Handedness.LEFT else 1
        self.jones_vector = np.array(
            [
                np.cos(tilt_angle) + sign * 1j * np.sin(tilt_angle) / axial_ratio,
                np.sin(tilt_angle) - sign * 1j * np.cos(tilt_angle) / axial_ratio,
            ]
        )
        # Normalize to unit magnitude.
        self.jones_vector /= np.linalg.norm(self.jones_vector)
        # Rotate such that the first element is real.
        self.jones_vector *= np.exp(-1j * np.angle(self.jones_vector[0]))

    @classmethod
    def lhcp(cls) -> typing.Self:
        """Left-hand circular polarization."""
        return cls(np.pi / 4 * u.rad, 1 * u.dimensionless, Handedness.LEFT)

    @classmethod
    def rhcp(cls) -> typing.Self:
        """Right-hand circular polarization."""
        return cls(np.pi / 4 * u.rad, 1 * u.dimensionless, Handedness.RIGHT)


class _ComplexInterpolator:
    """Interpolates complex values using log-magnitude and unit-phase components."""

    @enforce_units
    def __init__(
        self,
        theta: Angle,
        phi: Angle,
        frequency: Frequency | None,
        values: u.Quantity,
        floor: u.Quantity | None = None,
    ):
        r"""
        Create a complex interpolator.

        Parameters
        ----------
        theta: Angle
            1D array of equally spaced polar angles in [0, pi] radians with shape (N,).
        phi: Angle
            1D array of equally spaced azimuthal angles with shape (M,). The span should
            be at most 2π radians.
        frequency: Frequency | None
            Optional 1D array of strictly increasing frequencies with shape (K,).
        values: Quantity
            Complex array of values to interpolate. This should be in linear-scale (not
            log-scale like u.dB). The required shape depends on whether ``frequency`` is
            provided:
            - If ``frequency is None``: shape ``(N, M)``
            - Else: shape ``(N, M, K)``
        floor: Quantity | None
            Optional floor value for the magnitude in linear scale with the same unit as
            ``values``. If None, a default of ``1e-20 * values.unit`` is used. Magnitude
            interpolation is performed in log-space and thus cannot handle 0 values
            anywhere, so 0s are replaced with this prior to interpolation. Must be a
            positive scalar.
        """
        self.unit = values.unit
        values = values.value

        if floor is None:
            floor = 1e-20 * self.unit
        elif floor.unit != self.unit:
            raise ValueError("floor must be in the same unit as values.")
        if np.size(floor) != 1:
            raise ValueError("floor must be a scalar Quantity.")
        if floor.value <= 0:
            raise ValueError("floor must be > 0.")

        self._is_3d = frequency is not None

        phi = phi % (2 * np.pi * u.rad)

        # Sort phi and values to ensure phi is monotonically increasing after wrapping
        sort_indices = np.argsort(phi)
        phi = phi[sort_indices]
        values = values[:, sort_indices]

        # Remove duplicate phi values that can occur when input spans exactly 2π
        unique_indices = np.concatenate([[True], np.diff(phi) > 0])
        phi = phi[unique_indices]
        values = values[:, unique_indices]

        if frequency is None:
            values = values[:, :, np.newaxis]
            frequency = np.array([0.0]) * u.Hz  # Dummy frequency for 2D case

        delta_phi = np.diff(phi)[0]  # Assume equal spacing
        full_circle = delta_phi * phi.size >= 2 * np.pi * u.rad - (delta_phi / 2)

        if full_circle:
            # Pad phi with wrap-around columns so interpolation works at the 2π wrap
            # boundary.
            phi = np.concatenate(
                [
                    [phi[-1] - 2 * np.pi * u.rad],
                    phi,
                    [phi[0] + 2 * np.pi * u.rad],
                ]
            )
            values = np.concatenate(
                [values[:, -1:, :], values, values[:, :1, :]], axis=1
            )

        # Interpolate magnitude in log-space since this yields a more natural result
        # for antenna patterns.
        with np.errstate(divide="ignore", invalid="ignore"):
            mag_linear = np.clip(np.abs(values), floor.value, None)
            log_mag = np.log(mag_linear)

        # Interpolate phase as a unit-magnitude complex exponential. This avoids issues
        # with phase wrapping discontinuities.
        with np.errstate(invalid="ignore"):
            phase_exponential = np.where(
                np.abs(values) == 0,
                1.0,
                values / np.abs(values),
            )

        self._interp = scipy.interpolate.RegularGridInterpolator(
            points=(theta.value, phi.value, frequency.value),
            values=np.stack(
                [log_mag, np.real(phase_exponential), np.imag(phase_exponential)],
                axis=-1,
            ),
            method="linear",
            bounds_error=True,
        )

    @enforce_units
    def __call__(
        self, theta: Angle, phi: Angle, frequency: Frequency | None = None
    ) -> Dimensionless:
        r"""
        Interpolate at the given coordinates.

        Parameters
        ----------
        theta: Angle
            Polar angles.
        phi: Angle
            Azimuthal angles.
        frequency: Frequency | None
            Desired frequency. For 2D interpolators (constructed without ``frequency``),
            this argument is ignored.

        Returns
        -------
        Quantity
            Interpolated values. The unit will be the same as the unit of the values
            Quantity passed to the constructor. Shape is determined by standard Numpy
            broadcasting rules from the shapes of theta, phi, and frequency.

        Raises
        ------
        ValueError
            If phi, theta, or frequency are outside the range of the original grid.
        """
        if self._is_3d:
            if frequency is None:
                raise ValueError("Frequency must be provided for 3D interpolators")
        else:
            frequency = 0.0 * u.Hz

        points = (
            theta.value,
            (phi % (2 * np.pi * u.rad)).value,
            frequency.value,
        )

        features = self._interp(points)
        mag = np.exp(features[..., 0])
        phase_exp = features[..., 1] + 1j * features[..., 2]
        phase_exp /= np.abs(phase_exp)  # Re-normalize to remove numerical drift

        return mag * phase_exp * self.unit


class RadiationPattern:
    """Represents an antenna radiation pattern on a spherical coordinate system."""

    @enforce_units
    def __init__(
        self,
        theta: Angle,
        phi: Angle,
        frequency: Frequency | None,
        e_theta: Dimensionless,
        e_phi: Dimensionless,
        rad_efficiency: Dimensionless,
        default_polarization: Polarization | None = None,
        default_frequency: Frequency | None = None,
    ):
        r"""
        Create a radiation pattern from a set of E-field components.

        .. math::
            \vec{E}(\theta, \phi, f) = E_\theta(\theta, \phi, f)\hat{\theta}
            + E_\phi(\theta, \phi, f)\hat{\phi}

        Parameters
        ----------
        theta: Angle
            1D array of equally spaced polar angles in [0, pi] radians with shape (N,).
        phi: Angle
            1D array of equally spaced azimuthal angles with shape (M,). The span must
            be at most 2π radians.
        frequency: Frequency | None
            Optional 1D array of strictly increasing frequencies with shape (K,) where
            K > 1. If None, the pattern is treated as frequency-invariant and 2D over
            (theta, phi).
        e_theta: Dimensionless
            Complex array of :math:`E_{\theta}(\theta, \phi[, f])` values normalized
            such that the magnitude squared is equal to directivity. The required
            shape depends on whether ``frequency`` is provided:
            - If ``frequency is None``: shape ``(N, M)``
            - Else: shape ``(N, M, K)``
        e_phi: Dimensionless
            Complex array of :math:`E_{\phi}(\theta, \phi[, f])` values normalized
            such that the magnitude squared is equal to directivity. The required
            shape depends on whether ``frequency`` is provided:
            - If ``frequency is None``: shape ``(N, M)``
            - Else: shape ``(N, M, K)``
        rad_efficiency: Dimensionless
            Radiation efficiency :math:`\eta` in (0, 1].
        default_polarization: Polarization | None
            Optional default polarization used when instance methods are called without
            an explicit polarization.
        default_frequency: Frequency | None
            Optional default frequency used when instance methods are called without an
            explicit frequency.

        Raises
        ------
        ValueError
            If theta is not in [0, pi], if phi spans more than 2π radians, or if
            theta/phi/frequency are not strictly increasing, or if inputs are outside
            allowed ranges.
        ValueError
            If the surface integral of the directivity is greater than 4π.
        """
        self._validate_constructor_args(
            theta, phi, e_theta, e_phi, frequency, rad_efficiency
        )
        self.theta = theta
        self.phi = phi
        self.frequency = frequency
        self.e_theta = e_theta
        self.e_phi = e_phi
        self.rad_efficiency = rad_efficiency
        self.default_polarization = default_polarization
        self.default_frequency = default_frequency

        self._validate_surface_integral()

        self._e_theta_interp = _ComplexInterpolator(theta, phi, frequency, e_theta)
        self._e_phi_interp = _ComplexInterpolator(theta, phi, frequency, e_phi)

    @property
    def default_frequency(self) -> Frequency | None:
        """Default frequency used when instance methods are called without an explicit
        frequency.

        For frequency-aware patterns, this must be a scalar frequency within the closed
        interval defined by the pattern's ``frequency`` grid or None.

        For frequency-invariant patterns this is stored but ignored.
        """
        return getattr(self, "_default_frequency", None)

    @default_frequency.setter
    def default_frequency(self, value: Frequency | None) -> None:
        if value is None:
            self._default_frequency = None
            return

        if np.size(value) != 1:
            raise ValueError("default_frequency must be a scalar Quantity")

        # For 2D patterns (no frequency axis), store and ignore in computations
        if self.frequency is None:
            self._default_frequency = value
            return

        # For 3D patterns, enforce range within the frequency grid
        fmin = np.min(self.frequency)
        fmax = np.max(self.frequency)
        if not (fmin <= value <= fmax):
            raise ValueError(
                f"default_frequency {value} must be within [{fmin}, {fmax}]"
            )

        self._default_frequency = value

    @staticmethod
    def _validate_rad_efficiency(rad_efficiency: Dimensionless) -> None:
        """Validate that ``rad_efficiency`` is a scalar in the open interval (0, 1]."""
        if np.size(rad_efficiency) != 1:
            raise ValueError("rad_efficiency must be a scalar.")
        if not (0.0 < rad_efficiency.value <= 1.0):
            raise ValueError(f"rad_efficiency must be in (0, 1], got {rad_efficiency}.")

    def _validate_constructor_args(
        self,
        theta: Angle,
        phi: Angle,
        e_theta: Dimensionless,
        e_phi: Dimensionless,
        frequency: Frequency | None,
        rad_efficiency: Dimensionless,
    ):
        theta_step = np.diff(theta)
        phi_step = np.diff(phi)
        if not np.all(theta >= 0 * u.rad) or not np.all(theta <= np.pi * u.rad):
            raise ValueError("theta must be in [0, pi]")
        if not np.all(theta_step > 0 * u.rad):
            raise ValueError("theta must be strictly increasing")
        if not np.all(phi_step > 0 * u.rad):
            raise ValueError("phi must be strictly increasing")
        if theta_step.size > 0 and not np.allclose(theta_step, theta_step[0]):
            raise ValueError("theta must be equally spaced")
        if phi_step.size > 0 and not np.allclose(phi_step, phi_step[0]):
            raise ValueError("phi must be equally spaced")
        if (phi[-1] - phi[0]) > (2 * np.pi * u.rad + 0.5 * phi_step[0]):
            raise ValueError("phi must not span more than 2π radians")

        if frequency is not None:
            if np.size(frequency) < 2:
                raise ValueError("frequency must have length >= 2")
            if not np.all(np.diff(frequency) > 0 * u.Hz):
                raise ValueError("frequency must be strictly increasing")
            # Shape checks for 3D
            expected_shape = (theta.size, phi.size, frequency.size)
            if e_theta.shape != expected_shape or e_phi.shape != expected_shape:
                raise ValueError(
                    f"e_theta/e_phi must have shape {expected_shape}, "
                    f"got {e_theta.shape} and {e_phi.shape}"
                )
        else:
            # Shape checks for 2D
            expected_shape_2d = (theta.size, phi.size)
            if e_theta.shape != expected_shape_2d or e_phi.shape != expected_shape_2d:
                raise ValueError(
                    f"e_theta/e_phi must have shape {expected_shape_2d}, "
                    f"got {e_theta.shape} and {e_phi.shape}"
                )
        self._validate_rad_efficiency(rad_efficiency)

    def _validate_surface_integral(self):
        """Validate that surface integral of directivity is ≤ 4π."""
        total_directivity = np.abs(self.e_theta) ** 2 + np.abs(self.e_phi) ** 2

        if self.frequency is not None:
            for freq, freq_slice in zip(
                self.frequency, np.moveaxis(total_directivity, -1, 0), strict=False
            ):
                dir_surf_int = _surface_integral(self.theta, self.phi, freq_slice)
                if dir_surf_int > 1.01 * (4 * np.pi) * u.sr:
                    raise ValueError(
                        f"Surface integral of directivity {dir_surf_int} at {freq} is "
                        "greater than 4π."
                    )
        else:
            dir_surf_int = _surface_integral(self.theta, self.phi, total_directivity)
            if dir_surf_int > 1.01 * (4 * np.pi) * u.sr:
                raise ValueError(
                    f"Surface integral of directivity {dir_surf_int} is greater than "
                    "4π."
                )

    @classmethod
    @enforce_units
    def from_circular_e_field(
        cls,
        theta: Angle,
        phi: Angle,
        frequency: Frequency | None,
        e_lhcp: Dimensionless,
        e_rhcp: Dimensionless,
        rad_efficiency: Dimensionless,
        default_polarization: Polarization | None = None,
        default_frequency: Frequency | None = None,
    ) -> typing.Self:
        r"""
        Create a radiation pattern from a set of LHCP/RHCP E-field components.

        .. math::
            \vec{E}(\theta, \phi, f) = E_\text{LHCP}(\theta, \phi, f)\hat{\text{LHCP}}
            + E_\text{RHCP}(\theta, \phi, f)\hat{\text{RHCP}}

        Parameters
        ----------
        theta: Angle
            1D array of equally spaced polar angles in [0, pi] radians with shape (N,).
        phi: Angle
            1D array of equally spaced azimuthal angles with shape (M,). The span must
            be at most 2π radians.
        frequency: Frequency | None
            Optional 1D array of strictly increasing frequencies with shape (K,).
            If None, the pattern is treated as frequency-invariant and 2D over
            (theta, phi).
        e_lhcp: Dimensionless
            Complex array of :math:`E_{\text{LHCP}}(\theta, \phi[, f])` values
            normalized such that the magnitude squared is equal to directivity. The
            required shape depends on whether ``frequency`` is provided:
            - If ``frequency is None``: shape ``(N, M)``
            - Else: shape ``(N, M, K)``
        e_rhcp: Dimensionless
            Complex array of :math:`E_{\text{RHCP}}(\theta, \phi[, f])` values
            normalized such that the magnitude squared is equal to directivity. The
            required shape depends on whether ``frequency`` is provided:
            - If ``frequency is None``: shape ``(N, M)``
            - Else: shape ``(N, M, K)``
        rad_efficiency: Dimensionless
            Radiation efficiency :math:`\eta` in (0, 1].
        default_polarization: Polarization | None
            Optional default polarization used when instance methods are called without
            an explicit polarization.
        default_frequency: Frequency | None
            Optional default frequency used when instance methods are called without an
            explicit frequency.
        """
        # Change of basis from LHCP/RHCP to theta/phi.
        e_theta = 1 / np.sqrt(2) * (e_lhcp + e_rhcp)
        e_phi = 1j / np.sqrt(2) * (e_lhcp - e_rhcp)
        return cls(
            theta,
            phi,
            frequency,
            e_theta,
            e_phi,
            rad_efficiency,
            default_polarization,
            default_frequency,
        )

    @classmethod
    @enforce_units
    def from_circular_gain(
        cls,
        theta: Angle,
        phi: Angle,
        frequency: Frequency | None,
        gain_lhcp: Dimensionless,
        gain_rhcp: Dimensionless,
        phase_lhcp: Angle,
        phase_rhcp: Angle,
        rad_efficiency: Dimensionless,
        default_polarization: Polarization | None = None,
        default_frequency: Frequency | None = None,
    ) -> typing.Self:
        r"""
        Create a radiation pattern from circular gain and phase.

        Parameters
        ----------
        theta: Angle
            1D array of equally spaced polar angles in [0, pi] radians with shape (N,).
        phi: Angle
            1D array of equally spaced azimuthal angles with shape (M,). The span must
            be at most 2π radians.
        frequency: Frequency | None
            Optional 1D array of strictly increasing frequencies with shape (K,).
            If None, the pattern is treated as frequency-invariant and 2D over
            (theta, phi).
        gain_lhcp: Dimensionless
            Array of LHCP gain values. The required shape depends on whether
            ``frequency`` is provided:
            - If ``frequency is None``: shape ``(N, M)``
            - Else: shape ``(N, M, K)``
        gain_rhcp: Dimensionless
            Array of RHCP gain values. The required shape depends on whether
            ``frequency`` is provided:
            - If ``frequency is None``: shape ``(N, M)``
            - Else: shape ``(N, M, K)``
        phase_lhcp: Angle
            Array of LHCP phase angles. The required shape depends on whether
            ``frequency`` is provided:
            - If ``frequency is None``: shape ``(N, M)``
            - Else: shape ``(N, M, K)``
        phase_rhcp: Angle
            Array of RHCP phase angles. The required shape depends on whether
            ``frequency`` is provided:
            - If ``frequency is None``: shape ``(N, M)``
            - Else: shape ``(N, M, K)``
        rad_efficiency: Dimensionless
            Radiation efficiency :math:`\eta` in (0, 1].
        default_polarization: Polarization | None
            Optional default polarization used when instance methods are called without
            an explicit polarization.
        default_frequency: Frequency | None
            Optional default frequency used when instance methods are called without an
            explicit frequency.
        """
        if np.any(gain_lhcp < 0) or np.any(gain_rhcp < 0):
            raise ValueError("Gain must be non-negative")
        # Validate radiation efficiency prior to division
        cls._validate_rad_efficiency(rad_efficiency)

        e_lhcp = np.sqrt(gain_lhcp / rad_efficiency) * np.exp(1j * phase_lhcp.value)
        e_rhcp = np.sqrt(gain_rhcp / rad_efficiency) * np.exp(1j * phase_rhcp.value)
        return cls.from_circular_e_field(
            theta,
            phi,
            frequency,
            e_lhcp,
            e_rhcp,
            rad_efficiency,
            default_polarization,
            default_frequency,
        )

    @classmethod
    @enforce_units
    def from_linear_gain(
        cls,
        theta: Angle,
        phi: Angle,
        frequency: Frequency | None,
        gain_theta: Dimensionless,
        gain_phi: Dimensionless,
        phase_theta: Angle,
        phase_phi: Angle,
        rad_efficiency: Dimensionless,
        default_polarization: Polarization | None = None,
        default_frequency: Frequency | None = None,
    ) -> typing.Self:
        r"""
        Create a radiation pattern from linear gain and phase.

        Parameters
        ----------
        theta: Angle
            1D array of equally spaced polar angles in [0, pi] radians with shape (N,).
        phi: Angle
            1D array of equally spaced azimuthal angles with shape (M,). The span must
            be at most 2π radians.
        frequency: Frequency | None
            Optional 1D array of strictly increasing frequencies with shape (K,).
            If None, the pattern is treated as frequency-invariant and 2D over
            (theta, phi).
        gain_theta: Dimensionless
            Array of :math:`\hat{\theta}` gain values. The required shape depends on
            whether ``frequency`` is provided:
            - If ``frequency is None``: shape ``(N, M)``
            - Else: shape ``(N, M, K)``
        gain_phi: Dimensionless
            Array of :math:`\hat{\phi}` gain values. The required shape depends on
            whether ``frequency`` is provided:
            - If ``frequency is None``: shape ``(N, M)``
            - Else: shape ``(N, M, K)``
        phase_theta: Angle
            Array of :math:`\hat{\theta}` phase angles. The required shape depends on
            whether ``frequency`` is provided:
            - If ``frequency is None``: shape ``(N, M)``
            - Else: shape ``(N, M, K)``
        phase_phi: Angle
            Array of :math:`\hat{\phi}` phase angles. The required shape depends on
            whether ``frequency`` is provided:
            - If ``frequency is None``: shape ``(N, M)``
            - Else: shape ``(N, M, K)``
        rad_efficiency: Dimensionless
            Radiation efficiency :math:`\eta` in (0, 1].
        default_polarization: Polarization | None
            Optional default polarization used when instance methods are called without
            an explicit polarization.
        default_frequency: Frequency | None
            Optional default frequency used when instance methods are called without an
            explicit frequency.
        """
        if np.any(gain_theta < 0) or np.any(gain_phi < 0):
            raise ValueError("Gain must be non-negative")
        # Validate radiation efficiency prior to division
        cls._validate_rad_efficiency(rad_efficiency)

        e_theta = np.sqrt(gain_theta / rad_efficiency) * np.exp(1j * phase_theta.value)
        e_phi = np.sqrt(gain_phi / rad_efficiency) * np.exp(1j * phase_phi.value)
        return cls(
            theta,
            phi,
            frequency,
            e_theta,
            e_phi,
            rad_efficiency,
            default_polarization,
            default_frequency,
        )

    @enforce_units
    def e_field(
        self,
        theta: Angle,
        phi: Angle,
        *,
        frequency: Frequency | None = None,
        polarization: Polarization | None = None,
    ) -> Dimensionless:
        r"""
        Normalized complex E-field in the desired polarization state.

        Parameters
        ----------
        theta: Angle
            Polar angles.
        phi: Angle
            Azimuthal angles.
        frequency: Frequency | None
            Desired frequency. If None, uses the instance's ``default_frequency`` if
            set; otherwise raises ValueError. For 2D patterns (constructed without
            ``frequency``), this argument is ignored.
        polarization: Polarization | None
            Desired polarization state. If None, uses the instance's
            ``default_polarization`` if set; otherwise raises ValueError.

        Returns
        -------
        Dimensionless
            Complex E-field values. The E-field is normalized such that the magnitude
            squared is the directivity. Shape is determined by standard Numpy
            broadcasting rules from the shapes of theta and phi.
        """
        pol = polarization if polarization is not None else self.default_polarization
        if pol is None:
            raise ValueError(
                "Polarization must be provided or a default_polarization must be set "
                "on the RadiationPattern."
            )
        # Get frequency (None for 2D patterns, actual frequency for 3D)
        freq = frequency if frequency is not None else self.default_frequency
        if self.frequency is not None and freq is None:
            raise ValueError(
                "Frequency must be provided or a default_frequency must be set on the "
                "RadiationPattern."
            )
        e_jones = self._e_jones(theta, phi, freq)
        return (
            np.tensordot(pol.jones_vector.conj(), e_jones, axes=([-1], [-1]))
            * u.dimensionless
        )

    @enforce_units
    def directivity(
        self,
        theta: Angle,
        phi: Angle,
        *,
        frequency: Frequency | None = None,
        polarization: Polarization | None = None,
    ) -> Decibels:
        r"""
        Directivity of the antenna.

        Directivity as a function of the E-field in V/m is

        .. math::
            D(\theta, \phi) =
            \frac{ 4 \pi r^2 |\vec{E}(r, \theta, \phi)|^2 }{2\eta_0 P_\text{rad}}

        However, this class uses normalized E-fields since the intent is to represent
        only the relative power and phase of the E-field as a function of direction.
        Thus the directivity is simply

        .. math::
            D(\theta, \phi) = |\vec{E}(\theta, \phi)|^2


        Parameters
        ----------
        theta: Angle
            Polar angles.
        phi: Angle
            Azimuthal angles.
        frequency: Frequency | None
            Desired frequency. If None, uses the instance's ``default_frequency`` if
            set; otherwise raises ValueError. For 2D patterns (constructed without
            ``frequency``), this argument is ignored.
        polarization: Polarization | None
            Desired polarization state. If None, uses the instance's
            ``default_polarization`` if set; otherwise raises ValueError.

        Returns
        -------
        Decibels
            Directivity. Shape is determined by standard Numpy broadcasting rules from
            the shapes of theta and phi.
        """
        return to_dB(
            np.abs(
                self.e_field(theta, phi, frequency=frequency, polarization=polarization)
            )
            ** 2
        )

    @enforce_units
    def gain(
        self,
        theta: Angle,
        phi: Angle,
        *,
        frequency: Frequency | None = None,
        polarization: Polarization | None = None,
    ) -> Decibels:
        r"""
        Gain of the antenna.

        .. math::
            G(\theta, \phi) = \eta \cdot D(\theta, \phi)

        where :math:`\eta` is the radiation efficiency.

        Parameters
        ----------
        theta: Angle
            Polar angles.
        phi: Angle
            Azimuthal angles.
        frequency: Frequency | None
            Desired frequency. If None, uses the instance's ``default_frequency`` if
            set; otherwise raises ValueError. For 2D patterns (constructed without
            ``frequency``), this argument is ignored.
        polarization: Polarization | None
            Desired polarization state. If None, uses the instance's
            `default_polarization` if set; otherwise raises ValueError.

        Returns
        -------
        Decibels
            Gain. Shape is determined by standard Numpy broadcasting rules from the
            shapes of theta and phi.
        """
        return to_dB(self.rad_efficiency) + self.directivity(
            theta, phi, frequency=frequency, polarization=polarization
        )

    @enforce_units
    def axial_ratio(
        self,
        theta: Angle,
        phi: Angle,
        *,
        frequency: Frequency | None = None,
    ) -> Decibels:
        r"""
        Axial ratio of the antenna.

        The axial ratio is the ratio of the major to minor axis of the polarization
        ellipse. An axial ratio of 0 dB corresponds to circular polarization, and an
        axial ratio of ∞ corresponds to linear polarization. Elliptical polarizations
        have axial ratios between these two extremes.

        Parameters
        ----------
        theta: Angle
            Polar angles.
        phi: Angle
            Azimuthal angles.
        frequency: Frequency | None
            Desired frequency. If None, uses the instance's ``default_frequency`` if
            set; otherwise raises ValueError. For 2D patterns (constructed without
            ``frequency``), this argument is ignored.

        Returns
        -------
        Decibels
            Axial ratio. Shape is determined by standard Numpy broadcasting rules from
            the shapes of theta and phi.
        """
        # Get frequency (None for 2D patterns, actual frequency for 3D)
        freq = frequency if frequency is not None else self.default_frequency
        if self.frequency is not None and freq is None:
            raise ValueError(
                "Frequency must be provided or a default_frequency must be set on the "
                "RadiationPattern."
            )
        e_jones = self._e_jones(theta, phi, freq)
        coherency_matrix = np.einsum("...i,...j->...ij", e_jones, e_jones.conj())
        eigvals = np.linalg.eigvalsh(coherency_matrix.real)
        lambda_min = eigvals[..., 0]
        lambda_max = eigvals[..., 1]
        # Suppress divide-by-zero warnings.
        with np.errstate(divide="ignore"):
            return to_dB(np.sqrt(lambda_max / lambda_min) * u.dimensionless)

    @enforce_units
    def _e_jones(
        self, theta: Angle, phi: Angle, frequency: Frequency | None
    ) -> Dimensionless:
        e_theta_vals = self._e_theta_interp(theta, phi, frequency)
        e_phi_vals = self._e_phi_interp(theta, phi, frequency)
        return np.stack([e_theta_vals, e_phi_vals], axis=-1)


@enforce_units
def _surface_integral(theta: Angle, phi: Angle, values: Dimensionless) -> SolidAngle:
    r"""
    Take surface integral over a spherical surface.

    .. math::
        \int_\phi \int_\theta f(\theta, \phi) \sin(\theta) d\theta d\phi

    Parameters
    ----------
    theta: Angle
        1D array of equally spaced polar angles with shape (N,).
    phi: Angle
        1D array of equally spaced azimuthal angles with shape (M,).
    values:
        A 2D array with shape (N, M) giving the values to be integrated.

    Returns
    -------
        The result of the surface integral.
    """
    integrand = values * np.sin(theta[:, np.newaxis])
    int_phi = scipy.integrate.simpson(integrand, phi, axis=1)
    return scipy.integrate.simpson(int_phi, theta) * u.sr


@enforce_units
def gain_from_g_over_t(
    g_over_t: DecibelPerKelvin, temperature: Temperature
) -> Decibels:
    r"""
    Antenna gain from G/T and system noise temperature

    Parameters
    ----------
    g_over_t: DecibelPerKelvin
        Ratio of gain to the noise (G/T in dB/K)
    temperature: Temperature
        System noise temperature

    Returns
    -------
    Decibels
        Gain in dB.
    """
    if temperature < 0 * u.K:
        raise ValueError("Temperature must be positive")

    gain = g_over_t + temperature.to(u.dBK)
    return gain.to(u.dB)


@enforce_units
def temperature_from_g_over_t(
    g_over_t: DecibelPerKelvin, gain: Decibels
) -> Temperature:
    r"""
    Calculate the temperature in Kelvin from ratio of gain to the noise.

    Parameters
    ----------
    g_over_t: DecibelPerKelvin
        Ratio of gain to the noise.
    gain: Decibels
        Gain in dB.

    Returns
    -------
    Temperature
        System noise temperature
    """
    return (gain - g_over_t).to(u.K)
