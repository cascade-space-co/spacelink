r"""
Cassegrain Dual-Reflector Antenna Design
-----------------------------------------

Based on Granet (1998), "Designing Axially Symmetric Cassegrain or Gregorian
Dual-Reflector Antennas from Combinations of Prescribed Geometric Parameters,"
*IEEE Antennas and Propagation Magazine*, Vol. 40, No. 2, April 1998.

**Scope:** Cassegrain only (σ = −1), minimum-blockage condition, feed position
specified via ``Lm``. Corresponds to Table 2, Set No. 1 of the paper.

Coordinate System
-----------------

Origin at the paraboloid focus. z-axis points toward the subreflector
(positive z toward the feed side)::

         Dm
    |---------|
    |         |           z
    |   main  | vertex    |-->
    | reflector  apex     |
    |    at z=-F    feed  |  subrefl
    |         |    phase  |  apex
    |         |    center |  (z=a-f)
    |_________|  (z=-2f)  |
                          |
         <-- Lm ---->|<-- Ls ------>|
         (main apex  (feed phase    (feed to
          to feed)    centre        subrefl
                      to subrefl)   apex)

- Main reflector vertex at ``z = -F``
- Paraboloid focus (origin) at ``z = 0``
- Subreflector apex at ``z = a - f``
- Feed phase centre at ``z = -2f``
"""

import dataclasses

import astropy.units as u
import numpy as np

from .units import (
    Angle,
    Decibels,
    Dimensionless,
    Frequency,
    Length,
    enforce_units,
    wavelength,
)

__all__ = [
    "CassegrainGeometry",
    "main_reflector_profile",
    "subreflector_profile",
    "design_min_blockage",
    "cassegrain_gain",
]


@enforce_units
@dataclasses.dataclass(frozen=True)
class CassegrainGeometry:
    """Geometric parameters of a Cassegrain dual-reflector antenna.

    All lengths are stored in metres and the angle in radians, regardless of
    the units used at construction time (``@enforce_units`` handles conversion).

    Parameters
    ----------
    Dm : Length
        Main reflector diameter.
    F : Length
        Focal length of the main paraboloid.
    Lm : Length
        Axial distance from the main reflector apex to the feed phase centre.
    Ds : Length
        Subreflector diameter.
    Ls : Length
        Axial distance from the subreflector apex to the feed phase centre.
    a : Length
        Semi-transverse axis of the hyperboloid.
    f : Length
        Half-distance between the hyperboloid foci.
    theta_e : Angle
        Half-angle from the z-axis to the edge ray on the subreflector.
    """

    Dm: Length
    F: Length
    Lm: Length
    Ds: Length
    Ls: Length
    a: Length
    f: Length
    theta_e: Angle

    @property
    def eccentricity(self) -> u.Quantity:
        """Eccentricity of the hyperboloid, e = f/a > 1 for Cassegrain."""
        return (self.f / self.a).to(u.dimensionless_unscaled)

    @property
    def total_length(self) -> u.Quantity:
        """Total axial length from main reflector apex to feed phase centre (Lm + Ls)."""
        return self.Lm + self.Ls


@enforce_units
def main_reflector_profile(r: Length, F: Length) -> Length:
    r"""
    Axial coordinate of the main paraboloid at radial distance *r* (Eq. 1).

    .. math::

       z_{\text{mr}}(r) = \frac{r^2}{4F} - F

    Parameters
    ----------
    r : Length
        Radial distance from the axis (scalar or array, 0 ≤ r ≤ Dm/2).
    F : Length
        Focal length of the paraboloid.

    Returns
    -------
    Length
        z coordinate in metres. At r = 0: z = −F (vertex).
    """
    return r**2 / (4 * F) - F


@enforce_units
def subreflector_profile(r: Length, a: Length, f: Length) -> Length:
    r"""
    Axial coordinate of the hyperboloid subreflector at radial distance *r*
    (Eq. 2, Cassegrain: σ = −1).

    .. math::

       z_{\text{sr}}(r) = a \sqrt{1 + \frac{r^2}{f^2 - a^2}} - f

    Parameters
    ----------
    r : Length
        Radial distance from the axis (scalar or array, 0 ≤ r ≤ Ds/2).
    a : Length
        Semi-transverse axis of the hyperboloid.
    f : Length
        Half-distance between the hyperboloid foci (f > a for Cassegrain).

    Returns
    -------
    Length
        z coordinate in metres. At r = 0: z = a − f (subreflector apex).
    """
    b2 = f**2 - a**2
    if b2 <= 0 * u.m**2:
        raise ValueError(
            f"f² - a² = {b2:.4f} must be positive (requires f > a for Cassegrain)."
        )
    return a * np.sqrt(1 + r**2 / b2) - f


@enforce_units
def design_min_blockage(
    Dm: Length,
    F_over_D: Dimensionless,
    Lm: Length,
    Df: Length,
) -> CassegrainGeometry:
    """
    Design a minimum-blockage Cassegrain antenna from prescribed parameters.

    Implements Table 2, Set No. 1 (σ = −1) of Granet (1998), with the feed
    position specified via the axial distance *Lm*.

    Minimum-blockage condition (Hannan): F / (2f) = Ds / Df.

    Solve order::

        (0)   F       = F_over_D * Dm
        (i)   f       = (F - Lm) / 2                                      [Eq. 10]
        (ii)  Ds      = F * Df / (2 * f)                                   [Eq. 25]
        (iii) theta_e = arctan(8·F·Dm·Ds / (32·f·F·Dm - Ds·(16·F²-Dm²))) [Eq. 26]
        (iv)  Ls      = -2·Dm·f / (-Dm - 4·F·tan(theta_e/2))              [Eq. 27]
        (v)   a       = Ls - f                                             [Eq. 6]

    Parameters
    ----------
    Dm : Length
        Main reflector diameter.
    F_over_D : Dimensionless
        Focal ratio F/D (e.g. ``0.317 * u.dimensionless_unscaled``).
    Lm : Length
        Axial distance from the main reflector apex to the feed phase centre.
    Df : Length
        Overall feed aperture diameter (sets the minimum-blockage condition).

    Returns
    -------
    CassegrainGeometry

    Raises
    ------
    ValueError
        If the input combination does not yield a valid Cassegrain geometry.
    """
    sigma = -1

    # Step (0): derived focal length
    F = F_over_D * Dm
    if Lm <= 0 * u.m:
        raise ValueError(
            f"Lm = {Lm:.4f} must be positive (axial distance from main apex to feed)."
        )

    # Step (i): half-distance between hyperboloid foci — Eq. 10
    f = (F - Lm) / 2
    if f <= 0 * u.m:
        raise ValueError(
            f"f = (F - Lm) / 2 = {f:.4f} must be positive. "
            f"Ensure Lm < F (= {F:.4f})."
        )

    if Df <= 0 * u.m:
        raise ValueError(
            f"Df = {Df:.4f} must be positive (feed aperture diameter)."
        )

    # Step (ii): subreflector diameter — minimum-blockage condition, Eq. 25
    Ds = F * Df / (2 * f)
    if Ds >= Dm:
        raise ValueError(
            f"Subreflector diameter Ds = {Ds:.4f} must be less than main "
            f"reflector diameter Dm = {Dm:.4f}. Adjust Lm, F/D, or Df."
        )

    # Step (iii): subreflector edge half-angle — Eq. 26 (σ = −1)
    numerator = 8 * F * Dm * Ds
    denominator = 32 * f * F * Dm + sigma * Ds * (16 * F**2 - Dm**2)
    if denominator <= 0 * u.m**3:
        raise ValueError(
            f"theta_e formula denominator = {denominator:.4f} is non-positive. "
            f"The input parameters produce a degenerate geometry."
        )
    ratio = (numerator / denominator).to(u.dimensionless_unscaled)
    theta_e = np.arctan(ratio).to(u.rad)

    # Step (iv): subreflector-apex to feed distance — Eq. 27 (σ = −1)
    tan_half = np.tan(theta_e / 2).to(u.dimensionless_unscaled)
    Ls = 2 * sigma * Dm * f / (sigma * Dm - 4 * F * tan_half)
    if Ls <= 0 * u.m:
        raise ValueError(
            f"Ls = {Ls:.4f} must be positive. "
            f"The geometry is degenerate (subreflector-to-feed distance non-positive)."
        )

    # Step (v): semi-transverse axis — Eq. 6
    a = Ls - f
    if a <= 0 * u.m:
        raise ValueError(
            f"a = Ls - f = {a:.4f} must be positive. "
            f"The geometry is invalid for Cassegrain (a > 0 required)."
        )
    if a >= f:
        raise ValueError(
            f"a = {a:.4f} must be less than f = {f:.4f} for a valid Cassegrain "
            f"hyperboloid (eccentricity e = f/a > 1 required)."
        )

    return CassegrainGeometry(
        Dm=Dm, F=F, Lm=Lm, Ds=Ds, Ls=Ls, a=a, f=f, theta_e=theta_e
    )


@enforce_units
def cassegrain_gain(
    geometry: CassegrainGeometry,
    frequency: Frequency,
    efficiency: Dimensionless,
) -> Decibels:
    r"""
    Gain of a Cassegrain antenna in dBi (Eq. 3 of Granet (1998)).

    .. math::

       G = \eta \frac{\pi^2 (D_m^2 - D_s^2)}{\lambda^2}

    Parameters
    ----------
    geometry : CassegrainGeometry
        Antenna geometry, typically produced by :func:`design_min_blockage`.
    frequency : Frequency
        Operating frequency.
    efficiency : Dimensionless
        Overall antenna efficiency η (0 < η ≤ 1).

    Returns
    -------
    Decibels
        Gain in dBi.

    Raises
    ------
    ValueError
        If ``geometry.Ds >= geometry.Dm`` (non-positive effective aperture area).
    """
    if geometry.Ds >= geometry.Dm:
        raise ValueError(
            f"Ds = {geometry.Ds:.4f} >= Dm = {geometry.Dm:.4f}: "
            f"aperture area is non-positive."
        )
    _zero = 0 * u.dimensionless_unscaled
    _one = 1 * u.dimensionless_unscaled
    if not (_zero < efficiency <= _one):
        raise ValueError(
            f"efficiency = {efficiency} is out of range; must satisfy 0 < η ≤ 1."
        )

    wl = wavelength(frequency)
    gain_linear = efficiency * np.pi**2 * (geometry.Dm**2 - geometry.Ds**2) / wl**2
    return gain_linear.to(u.dB(1))
