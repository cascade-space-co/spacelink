r"""
Cassegrain Dual-Reflector Antenna Design
-----------------------------------------

Based on Granet (1998), "Designing Axially Symmetric Cassegrain or Gregorian
Dual-Reflector Antennas from Combinations of Prescribed Geometric Parameters,"
*IEEE Antennas and Propagation Magazine*, Vol. 40, No. 2, April 1998.

Kildal efficiency analysis based on Kildal (1983), "The Effects of Subreflector
Diffraction on the Aperture Efficiency of a Conventional Cassegrain Antenna — An
Analytical Approach," *IEEE Transactions on Antennas and Propagation*, Vol. AP-31,
No. 6, November 1983.

**Scope:** Cassegrain only (σ = −1), minimum-blockage condition, feed position
specified via ``Lm``. Corresponds to Table 2, Set No. 1 of Granet (1998).

Coordinate System
-----------------
Origin at the paraboloid focus. z-axis points toward the subreflector
(positive z toward the feed side)::

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
    "KildalEfficiency",
    # Granet profile functions
    "main_reflector_profile",
    "subreflector_profile",
    # Granet geometry primitives (Granet 1998, Table 2, Set No. 1)
    "hyperboloid_focus_half_distance",
    "subreflector_diameter_min_blockage",
    "subreflector_edge_angle",
    "subreflector_feed_distance",
    "hyperboloid_semi_axis",
    "main_rim_angle",
    # Kildal efficiency primitives (Kildal 1983)
    "kildal_feed_efficiency",
    "kildal_blockage_parameter",
    "kildal_diffraction_parameter",
    "kildal_blockage_term",
    "kildal_diffraction_term",
    "kildal_interference_efficiency",
    "kildal_optimal_ds_over_dm",
    # Design strategies
    "design_min_blockage",
    "design_kildal_optimum",
    # Performance functions
    "cassegrain_gain",
    "kildal_aperture_efficiency",
]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@enforce_units
@dataclasses.dataclass(frozen=True)
class CassegrainGeometry:
    """Geometric parameters of a Cassegrain dual-reflector antenna.

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
        """Total axial length from main reflector apex to feed phase centre."""
        return self.Lm + self.Ls


@dataclasses.dataclass(frozen=True)
class KildalEfficiency:
    """Aperture efficiency components from Kildal (1983) analysis.

    Combines feed efficiency with interference efficiency accounting for
    subreflector centre blockage and edge diffraction.

    Parameters
    ----------
    eta_f : Quantity[dimensionless]
        Feed (illumination) efficiency. Real-valued, 0 < η_f ≤ 1.
    delta_cb : Quantity[dimensionless]
        Centre-blockage interference term Δ_cb (Eq. 23). Real-valued, negative.
    delta_d : Quantity[dimensionless]
        Subreflector diffraction interference term Δ_d (Eq. 25). Complex-valued.
    eta_i : Quantity[dimensionless]
        Interference efficiency η_i = |1 + Δ_cb + Δ_d|² (Eq. 21). Real-valued.
    eta_a : Quantity[dimensionless]
        Total aperture efficiency η_a = η_f · η_i (Eq. 20). Real-valued.
    """

    eta_f: u.Quantity
    delta_cb: u.Quantity
    delta_d: u.Quantity
    eta_i: u.Quantity
    eta_a: u.Quantity


# ---------------------------------------------------------------------------
# Profile functions (Granet 1998, Eqs. 1–2)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Granet geometry primitives (Granet 1998, Table 2, Set No. 1, σ = −1)
# ---------------------------------------------------------------------------


@enforce_units
def hyperboloid_focus_half_distance(F: Length, Lm: Length) -> Length:
    r"""
    Half-distance between the hyperboloid foci (Granet 1998, Eq. 10).

    .. math::

       f = \frac{F - L_m}{2}

    Parameters
    ----------
    F : Length
        Focal length of the main paraboloid.
    Lm : Length
        Axial distance from the main reflector apex to the feed phase centre.

    Returns
    -------
    Length
        Half-distance *f* between the hyperboloid foci, in metres.

    Raises
    ------
    ValueError
        If ``Lm <= 0`` or ``Lm >= F`` (which would give f ≤ 0).
    """
    if Lm <= 0 * u.m:
        raise ValueError(
            f"Lm = {Lm:.4f} must be positive (axial distance from main apex to feed)."
        )
    f = (F - Lm) / 2
    if f <= 0 * u.m:
        raise ValueError(
            f"f = (F - Lm) / 2 = {f:.4f} must be positive. Ensure Lm < F (= {F:.4f})."
        )
    return f


@enforce_units
def subreflector_diameter_min_blockage(F: Length, Df: Length, f: Length) -> Length:
    r"""
    Subreflector diameter from the minimum-blockage condition (Granet 1998, Eq. 25).

    The Hannan minimum-blockage condition ``F / (2f) = Ds / Df`` gives:

    .. math::

       D_s = \frac{F \cdot D_f}{2f}

    Parameters
    ----------
    F : Length
        Focal length of the main paraboloid.
    Df : Length
        Overall feed aperture diameter.
    f : Length
        Half-distance between the hyperboloid foci.

    Returns
    -------
    Length
        Subreflector diameter *Ds*, in metres.

    Raises
    ------
    ValueError
        If ``Df <= 0``.
    """
    if Df <= 0 * u.m:
        raise ValueError(f"Df = {Df:.4f} must be positive (feed aperture diameter).")
    return F * Df / (2 * f)


@enforce_units
def subreflector_edge_angle(F: Length, Dm: Length, Ds: Length, f: Length) -> Angle:
    r"""
    Subreflector edge half-angle θ_e (Granet 1998, Eq. 26, σ = −1).

    .. math::

       \theta_e = \arctan\!\left(
           \frac{8 F D_m D_s}{32 f F D_m - D_s (16 F^2 - D_m^2)}
       \right)

    Parameters
    ----------
    F : Length
        Focal length of the main paraboloid.
    Dm : Length
        Main reflector diameter.
    Ds : Length
        Subreflector diameter.
    f : Length
        Half-distance between the hyperboloid foci.

    Returns
    -------
    Angle
        Subreflector edge half-angle in radians.

    Raises
    ------
    ValueError
        If the formula denominator is non-positive (degenerate geometry) or if
        ``Ds >= Dm``.
    """
    if Ds >= Dm:
        raise ValueError(
            f"Subreflector diameter Ds = {Ds:.4f} must be less than main "
            f"reflector diameter Dm = {Dm:.4f}."
        )
    sigma = -1
    numerator = 8 * F * Dm * Ds
    denominator = 32 * f * F * Dm + sigma * Ds * (16 * F**2 - Dm**2)
    if denominator <= 0 * u.m**3:
        raise ValueError(
            f"theta_e formula denominator = {denominator:.4f} is non-positive. "
            f"The input parameters produce a degenerate geometry."
        )
    ratio = (numerator / denominator).to(u.dimensionless_unscaled)
    return np.arctan(ratio).to(u.rad)


@enforce_units
def subreflector_feed_distance(
    Dm: Length, f: Length, F: Length, theta_e: Angle
) -> Length:
    r"""
    Axial distance from the subreflector apex to the feed phase centre
    (Granet 1998, Eq. 27, σ = −1).

    .. math::

       L_s = \frac{-2 D_m f}{-D_m - 4 F \tan(\theta_e / 2)}

    Parameters
    ----------
    Dm : Length
        Main reflector diameter.
    f : Length
        Half-distance between the hyperboloid foci.
    F : Length
        Focal length of the main paraboloid.
    theta_e : Angle
        Subreflector edge half-angle.

    Returns
    -------
    Length
        Distance *Ls* from subreflector apex to feed phase centre, in metres.

    Raises
    ------
    ValueError
        If the computed *Ls* is non-positive (degenerate geometry).
    """
    sigma = -1
    tan_half = np.tan(theta_e / 2).to(u.dimensionless_unscaled)
    Ls = 2 * sigma * Dm * f / (sigma * Dm - 4 * F * tan_half)
    if Ls <= 0 * u.m:
        raise ValueError(
            f"Ls = {Ls:.4f} must be positive. "
            f"The geometry is degenerate (subreflector-to-feed distance non-positive)."
        )
    return Ls


@enforce_units
def hyperboloid_semi_axis(Ls: Length, f: Length) -> Length:
    r"""
    Semi-transverse axis of the hyperboloid (Granet 1998, Eq. 6).

    .. math::

       a = L_s - f

    Parameters
    ----------
    Ls : Length
        Axial distance from the subreflector apex to the feed phase centre.
    f : Length
        Half-distance between the hyperboloid foci.

    Returns
    -------
    Length
        Semi-transverse axis *a*, in metres.

    Raises
    ------
    ValueError
        If ``a <= 0`` or ``a >= f`` (invalid Cassegrain hyperboloid).
    """
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
    return a


@enforce_units
def main_rim_angle(Dm: Length, F: Length) -> Angle:
    r"""
    Half-angle subtended by the main reflector rim at the paraboloid focus
    (Kildal 1983, Eq. 12).

    For a paraboloid: :math:`F = \frac{D}{4}\cot(\theta_0/2)`, which gives:

    .. math::

       \theta_0 = 2 \arctan\!\left(\frac{D_m}{4F}\right)

    Parameters
    ----------
    Dm : Length
        Main reflector diameter.
    F : Length
        Focal length of the main paraboloid.

    Returns
    -------
    Angle
        Main rim half-angle *θ₀* in radians.
    """
    ratio = (Dm / (4 * F)).to(u.dimensionless_unscaled)
    return (2 * np.arctan(ratio)).to(u.rad)


# ---------------------------------------------------------------------------
# Kildal efficiency primitives (Kildal 1983)
#
# These apply to the cos^n feed-pattern approximation A(ψ) = C(ψ) = cos^n(ψ/2),
# valid when the subreflector subtended angle ψ₀ = theta_e is small (< 30°).
# The user provides A₀ = cos^n(ψ₀/2) directly as the illumination taper at the
# subreflector edge (0 < A₀ ≤ 1).
#
# Notation mapping (Kildal → this module):
#   ψ₀  = theta_e  (subreflector edge half-angle)
#   θ₀  = theta_0  (main rim half-angle, from main_rim_angle())
#   d   = Ds       (subreflector diameter)
#   D   = Dm       (main reflector diameter)
# ---------------------------------------------------------------------------


@enforce_units
def kildal_feed_efficiency(A0: Dimensionless) -> Dimensionless:
    r"""
    Feed (illumination) efficiency for a cos^n feed pattern (Kildal 1983, Eq. 34).

    Valid in the limit ψ₀ → 0 (subreflector subtended angle small):

    .. math::

       \eta_f = \frac{2(1 - A_0)^2}{-\ln A_0}

    Parameters
    ----------
    A0 : Dimensionless
        Illumination taper at the subreflector edge: ``A₀ = cos^n(ψ₀/2)``.
        Must satisfy ``0 < A₀ < 1``.

    Returns
    -------
    Dimensionless
        Feed efficiency η_f.

    Raises
    ------
    ValueError
        If ``A0`` is not in the open interval (0, 1).
    """
    _zero = 0 * u.dimensionless_unscaled
    _one = 1 * u.dimensionless_unscaled
    if not (_zero < A0 < _one):
        raise ValueError(
            f"A0 = {A0} must satisfy 0 < A0 < 1 "
            f"(illumination taper at subreflector edge)."
        )
    a0 = A0.to(u.dimensionless_unscaled).value
    eta_f = 2 * (1 - a0) ** 2 / (-np.log(a0))
    return eta_f * u.dimensionless_unscaled


@enforce_units
def kildal_blockage_parameter(A0: Dimensionless) -> Dimensionless:
    r"""
    Blockage parameter *Cb* for a cos^n feed pattern (Kildal 1983, Eq. 35).

    Valid in the limit ψ₀ → 0:

    .. math::

       C_b = \frac{-\ln A_0}{1 - A_0}

    *Cb* depends only on the feed pattern taper and is independent of antenna
    size and subreflector diameter. It relates to the diffraction parameter via
    ``Cd ≈ Cb / π`` when ``θ₀ ≈ 90°`` (Kildal 1983, Eq. 29).

    Parameters
    ----------
    A0 : Dimensionless
        Illumination taper at the subreflector edge. Must satisfy ``0 < A₀ < 1``.

    Returns
    -------
    Dimensionless
        Blockage parameter *Cb*.

    Raises
    ------
    ValueError
        If ``A0`` is not in the open interval (0, 1).
    """
    _zero = 0 * u.dimensionless_unscaled
    _one = 1 * u.dimensionless_unscaled
    if not (_zero < A0 < _one):
        raise ValueError(
            f"A0 = {A0} must satisfy 0 < A0 < 1 "
            f"(illumination taper at subreflector edge)."
        )
    a0 = A0.to(u.dimensionless_unscaled).value
    Cb = -np.log(a0) / (1 - a0)
    return Cb * u.dimensionless_unscaled


@enforce_units
def kildal_diffraction_parameter(
    theta_0: Angle, theta_e: Angle, A0: Dimensionless
) -> Dimensionless:
    r"""
    Diffraction parameter *Cd* (Kildal 1983, Eq. 28).

    .. math::

       C_d = \frac{1}{\pi} \cdot
             \frac{\cos^2(\psi_0 / 2)}{\sqrt{\sin\theta_0}} \cdot C_b

    where *Cb* is computed from *A₀* via :func:`kildal_blockage_parameter` and
    ``ψ₀ = theta_e``.

    Parameters
    ----------
    theta_0 : Angle
        Main rim half-angle (from :func:`main_rim_angle`).
    theta_e : Angle
        Subreflector edge half-angle (ψ₀ in Kildal's notation).
    A0 : Dimensionless
        Illumination taper at the subreflector edge. Must satisfy ``0 < A₀ < 1``.

    Returns
    -------
    Dimensionless
        Diffraction parameter *Cd*.
    """
    Cb = kildal_blockage_parameter(A0)
    cos2_half_psi = (np.cos(theta_e / 2) ** 2).to(u.dimensionless_unscaled)
    sin_theta0 = np.sin(theta_0).to(u.dimensionless_unscaled)
    Cd = (1 / np.pi) * cos2_half_psi / np.sqrt(sin_theta0) * Cb
    return Cd.to(u.dimensionless_unscaled)


@enforce_units
def kildal_blockage_term(Cb: Dimensionless, Ds: Length, Dm: Length) -> Dimensionless:
    r"""
    Centre-blockage interference term Δ_cb (Kildal 1983, Eq. 23).

    .. math::

       \Delta_{cb} \approx -C_b \left(\frac{D_s}{D_m}\right)^2

    Valid when ``ψ_d ≪ ψ₀`` (blocked angle much smaller than subreflector edge
    angle), satisfied for most feeds when ``ψ₀ < 30°``.

    Parameters
    ----------
    Cb : Dimensionless
        Blockage parameter (from :func:`kildal_blockage_parameter`).
    Ds : Length
        Subreflector diameter.
    Dm : Length
        Main reflector diameter.

    Returns
    -------
    Dimensionless
        Centre-blockage term Δ_cb (real, negative).
    """
    d_over_D = (Ds / Dm).to(u.dimensionless_unscaled)
    return (-Cb * d_over_D**2).to(u.dimensionless_unscaled)


def kildal_diffraction_term(
    Cd: u.Quantity,
    Ds: u.Quantity,
    Dm: u.Quantity,
    frequency: u.Quantity,
    A0: u.Quantity,
) -> u.Quantity:
    r"""
    Subreflector diffraction interference term Δ_d (Kildal 1983, Eq. 25).

    .. math::

       \Delta_d = -(1 - j) C_d \sqrt{\frac{\lambda}{D_s}}
                  \sqrt{1 - \frac{D_s}{D_m}} A_0

    Parameters
    ----------
    Cd : Quantity[dimensionless]
        Diffraction parameter (from :func:`kildal_diffraction_parameter`).
    Ds : Quantity[length]
        Subreflector diameter.
    Dm : Quantity[length]
        Main reflector diameter.
    frequency : Quantity[frequency]
        Operating frequency.
    A0 : Quantity[dimensionless]
        Illumination taper at the subreflector edge.

    Returns
    -------
    Quantity[dimensionless]
        Diffraction term Δ_d (complex-valued).
    """
    lam = wavelength(frequency)
    cd = Cd.to(u.dimensionless_unscaled).value
    lam_over_d = (lam / Ds).to(u.dimensionless_unscaled).value
    d_over_D = (Ds / Dm).to(u.dimensionless_unscaled).value
    a0 = A0.to(u.dimensionless_unscaled).value
    delta_d = -(1 - 1j) * cd * np.sqrt(lam_over_d) * np.sqrt(1 - d_over_D) * a0
    return delta_d * u.dimensionless_unscaled


def kildal_interference_efficiency(
    delta_cb: u.Quantity,
    delta_d: u.Quantity,
) -> u.Quantity:
    r"""
    Interference efficiency η_i from blockage and diffraction (Kildal 1983, Eq. 21).

    .. math::

       \eta_i = |1 + \Delta_{cb} + \Delta_d|^2

    Parameters
    ----------
    delta_cb : Quantity[dimensionless]
        Centre-blockage term (from :func:`kildal_blockage_term`). Real-valued.
    delta_d : Quantity[dimensionless]
        Diffraction term (from :func:`kildal_diffraction_term`). Complex-valued.

    Returns
    -------
    Quantity[dimensionless]
        Interference efficiency η_i (real, non-negative).
    """
    total = (
        1 * u.dimensionless_unscaled
        + delta_cb.to(u.dimensionless_unscaled)
        + delta_d.to(u.dimensionless_unscaled)
    )
    return (np.abs(total.value) ** 2) * u.dimensionless_unscaled


@enforce_units
def kildal_optimal_ds_over_dm(
    theta_0: Angle,
    theta_e: Angle,
    A0: Dimensionless,
    Dm: Length,
    frequency: Frequency,
) -> Dimensionless:
    r"""
    Optimum subreflector-to-main-reflector diameter ratio (Kildal 1983, Eq. 31).

    Maximises the aperture efficiency by balancing centre-blockage loss against
    subreflector diffraction loss (neglecting support-strut blockage):

    .. math::

       \frac{D_s}{D_m} = \left[
           \frac{\cos^4(\psi_0/2)}{\sin\theta_0}
           \cdot \frac{A_0^2}{(4\pi)^2}
           \cdot \frac{\lambda}{D_m}
       \right]^{1/5}

    where ``ψ₀ = theta_e`` is the subreflector edge half-angle.

    Parameters
    ----------
    theta_0 : Angle
        Main rim half-angle (from :func:`main_rim_angle`).
    theta_e : Angle
        Subreflector edge half-angle (ψ₀ in Kildal's notation).
    A0 : Dimensionless
        Illumination taper at the subreflector edge. Must satisfy ``0 < A₀ < 1``.
    Dm : Length
        Main reflector diameter.
    frequency : Frequency
        Operating frequency.

    Returns
    -------
    Dimensionless
        Optimum ratio *Ds/Dm*.
    """
    lam = wavelength(frequency)
    cos4_half_psi = (np.cos(theta_e / 2) ** 4).to(u.dimensionless_unscaled)
    sin_theta0 = np.sin(theta_0).to(u.dimensionless_unscaled)
    lam_over_D = (lam / Dm).to(u.dimensionless_unscaled)
    a0_sq = (A0**2).to(u.dimensionless_unscaled)
    arg = cos4_half_psi / sin_theta0 * a0_sq * lam_over_D / (4 * np.pi) ** 2
    return (arg.to(u.dimensionless_unscaled) ** 0.2).to(u.dimensionless_unscaled)


# ---------------------------------------------------------------------------
# Design strategies
# ---------------------------------------------------------------------------


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
        (i)   f       = hyperboloid_focus_half_distance(F, Lm)         [Eq. 10]
        (ii)  Ds      = subreflector_diameter_min_blockage(F, Df, f)   [Eq. 25]
        (iii) theta_e = subreflector_edge_angle(F, Dm, Ds, f)          [Eq. 26]
        (iv)  Ls      = subreflector_feed_distance(Dm, f, F, theta_e)  [Eq. 27]
        (v)   a       = hyperboloid_semi_axis(Ls, f)                   [Eq. 6]

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
    F = F_over_D * Dm
    f = hyperboloid_focus_half_distance(F, Lm)
    Ds = subreflector_diameter_min_blockage(F, Df, f)
    theta_e = subreflector_edge_angle(F, Dm, Ds, f)
    Ls = subreflector_feed_distance(Dm, f, F, theta_e)
    a = hyperboloid_semi_axis(Ls, f)

    return CassegrainGeometry(
        Dm=Dm, F=F, Lm=Lm, Ds=Ds, Ls=Ls, a=a, f=f, theta_e=theta_e
    )


@enforce_units
def design_kildal_optimum(
    Dm: Length,
    F_over_D: Dimensionless,
    Lm: Length,
    frequency: Frequency,
    A0: Dimensionless,
) -> CassegrainGeometry:
    """
    Design a Cassegrain antenna with subreflector size optimised by Kildal (1983).

    Finds the subreflector diameter that maximises aperture efficiency by
    balancing centre-blockage loss against subreflector diffraction loss,
    using Kildal (1983) Eq. 31.

    A two-pass approach is used to handle the coupling between the optimum
    diameter and the subreflector edge angle *θ_e*:

    1. **Pass 1** — approximate ``cos⁴(ψ₀/2) ≈ 1`` (small-angle) to obtain
       an initial ``Ds₁``.
    2. **Pass 2** — compute ``θ_e`` for ``Ds₁``, then refine ``Ds`` using
       the exact form of Eq. 31.

    Parameters
    ----------
    Dm : Length
        Main reflector diameter.
    F_over_D : Dimensionless
        Focal ratio F/D.
    Lm : Length
        Axial distance from the main reflector apex to the feed phase centre.
    frequency : Frequency
        Operating frequency (determines wavelength for Eq. 31).
    A0 : Dimensionless
        Illumination taper at the subreflector edge: ``A₀ = cos^n(ψ₀/2)``.
        Must satisfy ``0 < A₀ < 1``.

    Returns
    -------
    CassegrainGeometry
        Geometry with *Ds* set to the Kildal-optimal value.

    Raises
    ------
    ValueError
        If the input combination does not yield a valid Cassegrain geometry,
        or if ``A0`` is out of range.
    """
    _zero = 0 * u.dimensionless_unscaled
    _one = 1 * u.dimensionless_unscaled
    if not (_zero < A0 < _one):
        raise ValueError(
            f"A0 = {A0} must satisfy 0 < A0 < 1 "
            f"(illumination taper at subreflector edge)."
        )

    F = F_over_D * Dm
    f = hyperboloid_focus_half_distance(F, Lm)
    theta_0 = main_rim_angle(Dm, F)

    # Pass 1: approximate theta_e = 0 → cos⁴(theta_e/2) ≈ 1
    theta_e_approx = 0 * u.rad
    ds_over_dm_1 = kildal_optimal_ds_over_dm(theta_0, theta_e_approx, A0, Dm, frequency)
    Ds1 = (ds_over_dm_1 * Dm).to(u.m)

    # Pass 2: compute theta_e for Ds1, then refine
    theta_e1 = subreflector_edge_angle(F, Dm, Ds1, f)
    ds_over_dm_2 = kildal_optimal_ds_over_dm(theta_0, theta_e1, A0, Dm, frequency)
    Ds2 = (ds_over_dm_2 * Dm).to(u.m)

    # Build the full geometry from the optimised Ds
    theta_e = subreflector_edge_angle(F, Dm, Ds2, f)
    Ls = subreflector_feed_distance(Dm, f, F, theta_e)
    a = hyperboloid_semi_axis(Ls, f)

    return CassegrainGeometry(
        Dm=Dm, F=F, Lm=Lm, Ds=Ds2, Ls=Ls, a=a, f=f, theta_e=theta_e
    )


# ---------------------------------------------------------------------------
# Performance functions
# ---------------------------------------------------------------------------


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


def kildal_aperture_efficiency(
    geometry: CassegrainGeometry,
    frequency: u.Quantity,
    A0: u.Quantity,
) -> KildalEfficiency:
    r"""
    Aperture efficiency of a Cassegrain antenna using Kildal (1983) analysis.

    Computes feed efficiency, centre-blockage, and diffraction terms, then
    combines them into the total aperture efficiency:

    .. math::

       \eta_a = \eta_f \cdot \eta_i, \quad
       \eta_i = |1 + \Delta_{cb} + \Delta_d|^2

    Valid for the cos^n feed pattern with small subreflector subtended angle
    (``theta_e < 30°``).

    Parameters
    ----------
    geometry : CassegrainGeometry
        Antenna geometry.
    frequency : Quantity[frequency]
        Operating frequency.
    A0 : Quantity[dimensionless]
        Illumination taper at the subreflector edge: ``A₀ = cos^n(ψ₀/2)``.
        Must satisfy ``0 < A₀ < 1``.

    Returns
    -------
    KildalEfficiency
        All intermediate efficiency components and the total η_a.
    """
    theta_0 = main_rim_angle(geometry.Dm, geometry.F)

    eta_f = kildal_feed_efficiency(A0)
    Cb = kildal_blockage_parameter(A0)
    Cd = kildal_diffraction_parameter(theta_0, geometry.theta_e, A0)
    delta_cb = kildal_blockage_term(Cb, geometry.Ds, geometry.Dm)
    delta_d = kildal_diffraction_term(Cd, geometry.Ds, geometry.Dm, frequency, A0)
    eta_i = kildal_interference_efficiency(delta_cb, delta_d)
    eta_a = (eta_f * eta_i).to(u.dimensionless_unscaled)

    return KildalEfficiency(
        eta_f=eta_f,
        delta_cb=delta_cb,
        delta_d=delta_d,
        eta_i=eta_i,
        eta_a=eta_a,
    )
