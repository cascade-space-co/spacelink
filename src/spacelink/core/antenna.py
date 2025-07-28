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
import functools
import typing

import astropy.units as u
import numpy as np
import scipy.integrate
import scipy.interpolate

from .units import (
    Angle,
    Decibels,
    Dimensionless,
    Frequency,
    Length,
    SolidAngle,
    wavelength,
    enforce_units,
    to_dB,
    to_linear,
    safe_negate,
)


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


class SphericalInterpolator:
    """Interpolates a regular grid of complex values in spherical coordinates."""

    @enforce_units
    def __init__(
        self,
        theta: Angle,
        phi: Angle,
        values: u.Quantity,
        floor: Decibels = -200 * u.dB,
    ):
        r"""
        Create a spherical interpolator.

        Parameters
        ----------
        theta: Angle
            1D array of equally spaced polar angles in [0, pi] radians with shape (N,).
        phi: Angle
            1D array of equally spaced azimuthal angles in [0, 2*pi) radians with shape
            (M,). Note that the last element must be less than 2π.
        values: u.Quantity
            2D complex array of values with shape (N, M) to interpolate.
        floor: Decibels
            Floor value for the magnitude in dB. The interpolation approach used cannot
            handle 0 values anywhere, so 0s (-∞ dB) are replaced with this prior to
            interpolation.
        """
        self.unit = values.unit

        phi_mod = phi % (2 * np.pi * u.rad)
        delta_phi = np.diff(phi_mod)[0]  # Assume equal spacing
        if np.isclose(delta_phi * phi.size, 2 * np.pi * u.rad):
            self.phi_min = 0 * u.rad
            self.phi_max = 2 * np.pi * u.rad
        else:  # phi does not span the full circle
            self.phi_min = np.min(phi % (2 * np.pi * u.rad))
            self.phi_max = np.max(phi % (2 * np.pi * u.rad))

        self.theta_min = np.min(theta)
        self.theta_max = np.max(theta)

        # RectSphereBivariateSpline requires theta to be in the range (0, pi),
        # excluding the endpoints where spherical coordinates have singularities.
        phi_slice = phi.value
        theta_start = 1 if np.isclose(theta[0], 0 * u.rad, atol=1e-10) else 0
        theta_end = -1 if np.isclose(theta[-1], np.pi * u.rad, atol=1e-10) else None
        theta_slice = theta[theta_start:theta_end].value
        values_slice = values[theta_start:theta_end, :].value

        with np.errstate(divide="ignore"):
            values_db = 10 * np.log10(np.abs(values_slice))
        # Apply a floor because spline interpolation can't handle -inf values anywhere.
        values_db = np.clip(values_db, floor.value, None)

        # Interpolate magnitude in log-space to avoid numerical stability issues that
        # can arise when interpolating magnitude or real and imaginary parts separately.
        self.log_mag = functools.partial(
            scipy.interpolate.RectSphereBivariateSpline(
                theta_slice,
                phi_slice,
                values_db,
            ),
            grid=False,
        )

        # Interpolate phase as a unit-magnitude complex exponential. This avoids issues
        # with phase wrapping discontinuities.
        with np.errstate(invalid="ignore"):
            phase_exponential = np.where(
                np.abs(values_slice) == 0,
                1.0,
                values_slice / np.abs(values_slice),
            )
        self.phase_real = functools.partial(
            scipy.interpolate.RectSphereBivariateSpline(
                theta_slice,
                phi_slice,
                np.real(phase_exponential),
            ),
            grid=False,
        )
        self.phase_imag = functools.partial(
            scipy.interpolate.RectSphereBivariateSpline(
                theta_slice,
                phi_slice,
                np.imag(phase_exponential),
            ),
            grid=False,
        )

    @enforce_units
    def __call__(self, theta: Angle, phi: Angle) -> u.Quantity:
        r"""
        Interpolate at the given spherical coordinates.

        Interpolate at points `(theta[i], phi[i]), i=0, ..., len(x)-1`. Standard Numpy
        broadcasting is obeyed.

        Parameters
        ----------
        theta: Angle
            Polar angles.
        phi: Angle
            Azimuthal angles with the same shape as theta.

        Returns
        -------
        u.Quantity
            Interpolated values. The unit will be the same as the unit of the values
            Quantity passed to the constructor.

        Raises
        ------
        ValueError
            If phi or theta are outside the range of the original grid.
        """
        phi_mod = phi % (2 * np.pi * u.rad)
        if np.any(phi_mod < self.phi_min) or np.any(phi_mod > self.phi_max):
            raise ValueError(f"phi must be in [{self.phi_min}, {self.phi_max}]")
        if np.any(theta < self.theta_min) or np.any(theta > self.theta_max):
            raise ValueError(f"theta must be in [{self.theta_min}, {self.theta_max}]")

        mag = 10 ** (self.log_mag(theta.value, phi.value) / 10)
        phase_exp = self.phase_real(theta, phi) + 1j * self.phase_imag(theta, phi)
        phase_exp /= np.abs(phase_exp)  # Re-normalize to remove numerical drift.
        return mag * phase_exp * self.unit


class RadiationPattern:
    """Represents an antenna radiation pattern on a spherical coordinate system."""

    @enforce_units
    def __init__(
        self,
        theta: Angle,
        phi: Angle,
        e_theta: Dimensionless,
        e_phi: Dimensionless,
        rad_efficiency: Dimensionless,
    ):
        r"""
        Create a radiation pattern from a set of E-field components.

        .. math::
            \vec{E}(\theta, \phi) = E_\theta(\theta, \phi)\hat{\theta}
            + E_\phi(\theta, \phi)\hat{\phi}

        Parameters
        ----------
        theta: Angle
            1D array of equally spaced polar angles in [0, pi] radians with shape (N,).
        phi: Angle
            1D array of equally spaced azimuthal angles in [0, 2*pi) radians with shape
            (M,). Note that the last element must be less than 2π.
        e_theta: Dimensionless
            2D complex array of :math:`E_{\theta}(\theta, \phi)` values with shape (N,
            M) normalized such that the magnitude squared is equal to directivity.
        e_phi: Dimensionless
            2D complex array of :math:`E_{\phi}(\theta, \phi)` values with shape (N, M)
            normalized such that the magnitude squared is equal to directivity.
        rad_efficiency: Dimensionless
            Radiation efficiency :math:`\eta` in [0, 1].
        """
        self.theta = theta
        self.phi = phi
        self.e_theta = e_theta
        self.e_phi = e_phi
        self.rad_efficiency = rad_efficiency

        # Surface integral of directivity should be 4π over the whole sphere (or less if
        # the pattern is not defined over the whole sphere). It should never be greater
        # than 4π.
        dir_surf_int = _surface_integral(
            theta, phi, np.abs(e_theta) ** 2 + np.abs(e_phi) ** 2
        )
        if dir_surf_int > 1.01 * (4 * np.pi) * u.sr:
            raise ValueError(
                f"Surface integral of directivity {dir_surf_int} is greater than 4π."
            )

        self.e_theta_interp = SphericalInterpolator(theta, phi, e_theta)
        self.e_phi_interp = SphericalInterpolator(theta, phi, e_phi)

    @classmethod
    @enforce_units
    def from_circular_e_field(
        cls,
        theta: Angle,
        phi: Angle,
        e_lhcp: Dimensionless,
        e_rhcp: Dimensionless,
        rad_efficiency: Dimensionless,
    ) -> typing.Self:
        r"""
        Create a radiation pattern from a set of LHCP/RHCP E-field components.

        .. math::
            \vec{E}(\theta, \phi) = E_\text{LHCP}(\theta, \phi)\hat{\text{LHCP}}
            + E_\text{RHCP}(\theta, \phi)\hat{\text{RHCP}}

        Parameters
        ----------
        theta: Angle
            1D array of equally spaced polar angles in [0, pi] radians with shape (N,).
        phi: Angle
            1D array of equally spaced azimuthal angles in [0, 2*pi) radians with shape
            (M,).
        e_lhcp: Dimensionless
            2D complex array of :math:`E_{\text{LHCP}}(\theta, \phi)` values with shape
            (N, M) normalized such that the magnitude squared is equal to directivity.
        e_rhcp: Dimensionless
            2D complex array of :math:`E_{\text{RHCP}}(\theta, \phi)` values with shape
            (N, M) normalized such that the magnitude squared is equal to directivity.
        rad_efficiency: Dimensionless
            Radiation efficiency :math:`\eta` in [0, 1].
        """
        # Change of basis from LHCP/RHCP to theta/phi.
        e_theta = 1 / np.sqrt(2) * (e_lhcp + e_rhcp)
        e_phi = 1j / np.sqrt(2) * (e_lhcp - e_rhcp)
        return cls(theta, phi, e_theta, e_phi, rad_efficiency)

    @classmethod
    @enforce_units
    def from_circular_gain(
        cls,
        theta: Angle,
        phi: Angle,
        gain_lhcp: Dimensionless,
        gain_rhcp: Dimensionless,
        phase_lhcp: Angle,
        phase_rhcp: Angle,
        rad_efficiency: Dimensionless,
    ) -> typing.Self:
        r"""
        Create a radiation pattern from circular gain and phase.

        Parameters
        ----------
        theta: Angle
            1D array of equally spaced polar angles in [0, pi] radians with shape (N,).
        phi: Angle
            1D array of equally spaced azimuthal angles in [0, 2*pi) radians with shape
            (M,).
        gain_lhcp: Dimensionless
            2D array of LHCP gain with shape (N, M).
        gain_rhcp: Dimensionless
            2D array of RHCP gain with shape (N, M).
        phase_lhcp: Angle
            2D array of LHCP phase angles with shape (N, M).
        phase_rhcp: Angle
            2D array of RHCP phase angles with shape (N, M).
        rad_efficiency: Dimensionless
            Radiation efficiency :math:`\eta` in [0, 1].
        """
        if np.any(gain_lhcp < 0) or np.any(gain_rhcp < 0):
            raise ValueError("Gain must be non-negative")

        e_lhcp = np.sqrt(gain_lhcp / rad_efficiency) * np.exp(1j * phase_lhcp.value)
        e_rhcp = np.sqrt(gain_rhcp / rad_efficiency) * np.exp(1j * phase_rhcp.value)
        return cls.from_circular_e_field(theta, phi, e_lhcp, e_rhcp, rad_efficiency)

    @classmethod
    @enforce_units
    def from_linear_gain(
        cls,
        theta: Angle,
        phi: Angle,
        gain_theta: Dimensionless,
        gain_phi: Dimensionless,
        phase_theta: Angle,
        phase_phi: Angle,
        rad_efficiency: Dimensionless,
    ) -> typing.Self:
        r"""
        Create a radiation pattern from linear gain and phase.

        Parameters
        ----------
        theta: Angle
            1D array of equally spaced polar angles in [0, pi] radians with shape (N,).
        phi: Angle
            1D array of equally spaced azimuthal angles in [0, 2*pi) radians with shape
            (M,).
        gain_theta: Dimensionless
            2D array of :math:`\hat{\theta}` gain with shape (N, M).
        gain_phi: Dimensionless
            2D array of :math:`\hat{\phi}` gain with shape (N, M).
        phase_theta: Angle
            2D array of :math:`\hat{\theta}` phase angles with shape (N, M).
        phase_phi: Angle
            2D array of :math:`\hat{\phi}` phase angles with shape (N, M).
        rad_efficiency: Dimensionless
            Radiation efficiency :math:`\eta` in [0, 1].
        """
        if np.any(gain_theta < 0) or np.any(gain_phi < 0):
            raise ValueError("Gain must be non-negative")

        e_theta = np.sqrt(gain_theta / rad_efficiency) * np.exp(1j * phase_theta.value)
        e_phi = np.sqrt(gain_phi / rad_efficiency) * np.exp(1j * phase_phi.value)
        return cls(theta, phi, e_theta, e_phi, rad_efficiency)

    @enforce_units
    def e_field(
        self, theta: Angle, phi: Angle, polarization: Polarization
    ) -> Dimensionless:
        r"""
        Normalized complex E-field in the desired polarization state.

        Parameters
        ----------
        theta: Angle
            Polar angles.
        phi: Angle
            Azimuthal angles.
        polarization: Polarization
            Desired polarization state.

        Returns
        -------
        Dimensionless
            Complex E-field values. The E-field is normalized such that the magnitude
            squared is the directivity. Shape is determined by standard Numpy
            broadcasting rules from the shapes of theta and phi.
        """
        e_jones = self._e_jones(theta, phi)
        return (
            np.tensordot(polarization.jones_vector.conj(), e_jones, axes=([-1], [-1]))
            * u.dimensionless
        )

    @enforce_units
    def directivity(
        self, theta: Angle, phi: Angle, polarization: Polarization
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
        polarization: Polarization
            Desired polarization state.

        Returns
        -------
        Decibels
            Directivity. Shape is determined by standard Numpy broadcasting rules from
            the shapes of theta and phi.
        """
        return to_dB(np.abs(self.e_field(theta, phi, polarization)) ** 2)

    @enforce_units
    def gain(self, theta: Angle, phi: Angle, polarization: Polarization) -> Decibels:
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
        polarization: Polarization
            Desired polarization state.

        Returns
        -------
        Decibels
            Gain. Shape is determined by standard Numpy broadcasting rules from the
            shapes of theta and phi.
        """
        return to_dB(self.rad_efficiency) + self.directivity(theta, phi, polarization)

    @enforce_units
    def axial_ratio(self, theta: Angle, phi: Angle) -> Decibels:
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

        Returns
        -------
        Decibels
            Axial ratio. Shape is determined by standard Numpy broadcasting rules from
            the shapes of theta and phi.
        """
        e_jones = self._e_jones(theta, phi)
        coherency_matrix = np.einsum("...i,...j->...ij", e_jones, e_jones.conj())
        eigvals = np.linalg.eigvalsh(coherency_matrix.real)
        lambda_min = eigvals[..., 0]
        lambda_max = eigvals[..., 1]
        # Suppress divide-by-zero warnings.
        with np.errstate(divide="ignore"):
            return to_dB(np.sqrt(lambda_max / lambda_min) * u.dimensionless)

    @enforce_units
    def _e_jones(self, theta: Angle, phi: Angle) -> Dimensionless:
        e_theta = self.e_theta_interp(theta, phi)
        e_phi = self.e_phi_interp(theta, phi)
        return np.stack([e_theta, e_phi], axis=-1) * u.dimensionless


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
