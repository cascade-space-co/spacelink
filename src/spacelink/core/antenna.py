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
"""

import typing

import astropy.units as u
import numpy as np
import scipy.interpolate

from .units import (
    Angle,
    DecibelMilliwatts,
    Decibels,
    Dimensionless,
    Frequency,
    Length,
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


class Polarization:
    """Represents a polarization state."""

    @enforce_units
    def __init__(
        self,
        tilt_angle: Angle,
        axial_ratio: Dimensionless,
        phase_difference: Angle,
    ):
        r"""
        Create a polarization state.

        Parameters
        ----------
        tilt_angle: Angle
            Tilt angle of the polarization ellipse, measured in the local tangent plane,
            relative to :math:`\hat{\theta}`.
        axial_ratio: Dimensionless
            Axial ratio of the polarization ellipse.
        phase_difference: Angle
            Phase difference between the components along :math:`\hat{\theta}` and
            :math:`\hat{\phi}`.
        """
        self.tilt_angle = tilt_angle
        self.axial_ratio = axial_ratio
        self.phase_difference = phase_difference
        self.jones_vector = np.array(
            [
                np.cos(tilt_angle),
                np.exp(1j * phase_difference.value) * np.sin(tilt_angle) / axial_ratio,
            ]
        )
        self.jones_vector /= np.linalg.norm(self.jones_vector)

    @classmethod
    def lhcp(cls) -> typing.Self:
        """Left-hand circular polarization."""
        return cls(np.pi / 4 * u.rad, 1 * u.dimensionless, np.pi / 2 * u.rad)

    @classmethod
    def rhcp(cls) -> typing.Self:
        """Right-hand circular polarization."""
        return cls(np.pi / 4 * u.rad, 1 * u.dimensionless, -np.pi / 2 * u.rad)


class AntennaPattern:
    """Represents an antenna pattern on a spherical coordinate system."""

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
        Create an antenna pattern from a set of E-field components.

        .. math::
            \vec{E}(\theta, \phi) = E_\theta(\theta, \phi)\hat{\theta}
            + E_\phi(\theta, \phi)\hat{\phi}

        Parameters
        ----------
        theta: Angle
            1D array of polar angles in [0, pi] radians with shape (N,).
        phi: Angle
            1D array of azimuthal angles in [0, 2*pi) radians with shape (M,).
        e_theta: Dimensionless
            2D complex array of :math:`E_{\theta}(\theta, \phi)` values with shape [N,
            M] normalized such that the magnitude squared is equal to directivity.
        e_phi: Dimensionless
            2D complex array of :math:`E_{\phi}(\theta, \phi)` values with shape (N, M)
            normalized such that the magnitude squared is equal to directivity.
        rad_efficiency: Dimensionless
            Radiation efficiency :math:`\eta` in [0, 1].
        """
        self.phi = phi
        self.theta = theta
        self.e_theta = e_theta
        self.e_phi = e_phi
        self.rad_efficiency = rad_efficiency

        # Surface integral of directivity should be 4π over the whole sphere (or less if
        # the pattern is not defined over the whole sphere). It should never be greater
        # than 4π.
        dir_surf_int = self._surface_integral(np.abs(e_theta) ** 2 + np.abs(e_phi) ** 2)
        if dir_surf_int > 1.05 * (4 * np.pi):
            raise ValueError(
                f"Surface integral of directivity {dir_surf_int} is greater than 4π."
            )

        # Functions for interpolating the complex field. Unfortunately
        # RectSphereBivariateSpline doesn't support complex data, so we have to
        # interpolate the real and imaginary parts separately.
        # TODO: Only need to leave off first/last elements of theta if they are 0 and pi
        self.e_theta_real_interp = scipy.interpolate.RectSphereBivariateSpline(
            self.theta[1:-1],
            self.phi,
            np.real(e_theta[1:-1, :]),
        )
        self.e_theta_imag_interp = scipy.interpolate.RectSphereBivariateSpline(
            self.theta[1:-1],
            self.phi,
            np.imag(e_theta[1:-1, :]),
        )
        self.e_phi_real_interp = scipy.interpolate.RectSphereBivariateSpline(
            self.theta[1:-1],
            self.phi,
            np.real(e_phi[1:-1, :]),
        )
        self.e_phi_imag_interp = scipy.interpolate.RectSphereBivariateSpline(
            self.theta[1:-1],
            self.phi,
            np.imag(e_phi[1:-1, :]),
        )

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
        Create an antenna pattern from a set of LHCP/RHCP E-field components.

        .. math::
            \vec{E}(\theta, \phi) = E_\text{LHCP}(\theta, \phi)\hat{\text{LHCP}}
            + E_\text{RHCP}(\theta, \phi)\hat{\text{RHCP}}

        Parameters
        ----------
        theta: Angle
            1D array of polar angles in [0, pi] radians with shape (N,).
        phi: Angle
            1D array of azimuthal angles in [0, 2*pi) radians with shape (M,).
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
        Create an antenna pattern from circular gain and phase.

        Parameters
        ----------
        theta: Angle
            1D array of polar angles in [0, pi] radians with shape (N,).
        phi: Angle
            1D array of azimuthal angles in [0, 2*pi) radians with shape (M,).
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
        e_lhcp = np.sqrt(gain_lhcp / rad_efficiency) * np.exp(1j * phase_lhcp)
        e_rhcp = np.sqrt(gain_rhcp / rad_efficiency) * np.exp(1j * phase_rhcp)
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
        Create an antenna pattern from linear gain and phase.

        Parameters
        ----------
        theta: Angle
            1D array of polar angles in [0, pi] radians with shape (N,).
        phi: Angle
            1D array of azimuthal angles in [0, 2*pi) radians with shape (M,).
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
        e_theta = np.sqrt(gain_theta / rad_efficiency) * np.exp(1j * phase_theta)
        e_phi = np.sqrt(gain_phi / rad_efficiency) * np.exp(1j * phase_phi)
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
            Azimuthal angles with the same shape as theta.
        polarization: Polarization
            Desired polarization state.

        Returns
        -------
        Dimensionless
            Complex E-field values. The E-field is normalized such that the magnitude
            squared is the directivity. Shape is the same as theta and phi.
        """
        e_theta = self.e_theta_real_interp(theta, phi) + 1j * self.e_theta_imag_interp(
            theta, phi
        )
        e_phi = self.e_phi_real_interp(theta, phi) + 1j * self.e_phi_imag_interp(
            theta, phi
        )
        e_jones = np.stack([e_theta, e_phi])
        return (polarization.jones_vector.conj().T @ e_jones) * u.dimensionless

    @enforce_units
    def directivity(
        self, theta: Angle, phi: Angle, polarization: Polarization
    ) -> Dimensionless:
        r"""
        Directivity of the antenna.

        Directivity as a function of the E-field in V/m is

        .. math::
            D(\theta, \phi) = \frac{ 4 \pi r^2 |\vec{E}(r, \theta, \phi)|^2 }{2\eta_0 P_\text{rad}}

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
            Azimuthal angles with the same shape as theta.
        polarization: Polarization
            Desired polarization state.

        Returns
        -------
        Dimensionless
            Directivity with the same shape as theta and phi.
        """
        return np.abs(self.e_field(theta, phi, polarization)) ** 2

    @enforce_units
    def gain(
        self, theta: Angle, phi: Angle, polarization: Polarization
    ) -> Dimensionless:
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
            Azimuthal angles with the same shape as theta.
        polarization: Polarization
            Desired polarization state.

        Returns
        -------
        Dimensionless
            Directivity with the same shape as theta and phi.
        """
        return self.rad_efficiency * self.directivity(theta, phi, polarization)

    @enforce_units
    def axial_ratio(self, theta: Angle, phi: Angle) -> Decibels:
        r"""
        Axial ratio of the antenna.

        The axial ratio is the ratio of the major to minor axis of the polarization
        ellipse. An axial ratio of 0 dB corresponds to circular polarization, and an
        axial ratio of ∞ corresponds to linear polarization. Elliptical polarizations
        are found between these two extremes.

        Parameters
        ----------
        theta: Angle
            Polar angles.
        phi: Angle
            Azimuthal angles.

        Returns
        -------
        Decibels
            Axial ratios in decibels with the same shape as theta and phi.
        """
        mag_lhcp = np.abs(self.e_field(theta, phi, Polarization.lhcp()))
        mag_rhcp = np.abs(self.e_field(theta, phi, Polarization.rhcp()))
        # Suppress divide-by-zero warnings.
        with np.errstate(divide="ignore"):
            return to_dB(
                np.maximum(mag_lhcp, mag_rhcp)
                / np.minimum(mag_lhcp, mag_rhcp)
                * u.dimensionless
            )

    def _surface_integral(self, values: np.ndarray) -> float:
        r"""
        Take surface integral over the full pattern.

        If the pattern is defined over the full sphere then this will integrate over
        the full sphere. Otherwise this will integrate over the solid angle where the
        pattern is defined, as if the pattern had zero gain (-∞ dBi) in directions
        where it is not defined.

        Parameters
        ----------
        values:
            A 2D array giving the values to be integrated over the full sphere. Must
            have shape (N, M) where N is the size of theta and M is the size of phi as
            passed to the constructor.

        Returns
        -------
            The result of the surface integral.
        """
        delta_theta = np.diff(self.theta)[0]
        delta_phi = np.diff(self.phi)[0]
        rings = np.sum(values, axis=1) * delta_phi
        return np.sum(rings * np.sin(self.theta) * delta_theta)
