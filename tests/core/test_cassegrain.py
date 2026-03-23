"""Tests for the Cassegrain dual-reflector antenna design module."""

import astropy.units as u
import numpy as np
import pytest
from astropy.tests.helper import assert_quantity_allclose

from spacelink.core.cassegrain import (
    CassegrainGeometry,
    cassegrain_gain,
    design_min_blockage,
    main_reflector_profile,
    subreflector_profile,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ref_geometry() -> CassegrainGeometry:
    """Reference geometry from the Excel-verified test case in cassegrain_design.md."""
    return design_min_blockage(
        Dm=5 * u.m,
        F_over_D=0.317 * u.dimensionless_unscaled,
        Lm=1.1 * u.m,
        Df=0.1 * u.m,
    )


# ---------------------------------------------------------------------------
# design_min_blockage — reference case
# ---------------------------------------------------------------------------

def test_design_min_blockage_focal_length(ref_geometry: CassegrainGeometry) -> None:
    assert_quantity_allclose(ref_geometry.F, 1.585 * u.m, rtol=1e-4)


def test_design_min_blockage_f(ref_geometry: CassegrainGeometry) -> None:
    assert_quantity_allclose(ref_geometry.f, 0.2425 * u.m, rtol=1e-4)


def test_design_min_blockage_Ds(ref_geometry: CassegrainGeometry) -> None:
    assert_quantity_allclose(ref_geometry.Ds, 0.326804124 * u.m, rtol=1e-6)


def test_design_min_blockage_theta_e(ref_geometry: CassegrainGeometry) -> None:
    assert_quantity_allclose(ref_geometry.theta_e, 0.351304076 * u.rad, rtol=1e-5)


def test_design_min_blockage_Ls(ref_geometry: CassegrainGeometry) -> None:
    assert_quantity_allclose(ref_geometry.Ls, 0.395903483 * u.m, rtol=1e-6)


def test_design_min_blockage_a(ref_geometry: CassegrainGeometry) -> None:
    assert_quantity_allclose(ref_geometry.a, 0.153403483 * u.m, rtol=1e-6)


# ---------------------------------------------------------------------------
# CassegrainGeometry properties
# ---------------------------------------------------------------------------

def test_eccentricity(ref_geometry: CassegrainGeometry) -> None:
    """Eccentricity e = f/a must be > 1 (hyperboloid) and match reference."""
    e = ref_geometry.eccentricity
    assert e > 1.0 * u.dimensionless_unscaled
    assert_quantity_allclose(e, 1.580798528 * u.dimensionless_unscaled, rtol=1e-5)


def test_total_length(ref_geometry: CassegrainGeometry) -> None:
    """Total length Lm + Ls must match the reference value."""
    assert_quantity_allclose(ref_geometry.total_length, 1.495903483 * u.m, rtol=1e-6)


# ---------------------------------------------------------------------------
# main_reflector_profile
# ---------------------------------------------------------------------------

def test_main_reflector_profile_vertex() -> None:
    """At r = 0 the vertex is at z = -F."""
    F = 1.585 * u.m
    z = main_reflector_profile(0 * u.m, F)
    assert_quantity_allclose(z, -F)


def test_main_reflector_profile_array() -> None:
    """Array input: output shape must match input shape."""
    F = 1.585 * u.m
    r = np.linspace(0, 2.5, 10) * u.m
    z = main_reflector_profile(r, F)
    assert z.shape == r.shape
    # Vertex at index 0
    assert_quantity_allclose(z[0], -F)
    # Profile is monotonically increasing (less negative) away from vertex
    assert np.all(np.diff(z.value) >= 0)


# ---------------------------------------------------------------------------
# subreflector_profile
# ---------------------------------------------------------------------------

def test_subreflector_profile_vertex(ref_geometry: CassegrainGeometry) -> None:
    """At r = 0 the apex is at z = a - f."""
    z = subreflector_profile(0 * u.m, ref_geometry.a, ref_geometry.f)
    assert_quantity_allclose(z, ref_geometry.a - ref_geometry.f)


def test_subreflector_profile_array(ref_geometry: CassegrainGeometry) -> None:
    """Array input: output shape must match input shape."""
    r = np.linspace(0, ref_geometry.Ds.value / 2, 10) * u.m
    z = subreflector_profile(r, ref_geometry.a, ref_geometry.f)
    assert z.shape == r.shape
    # Apex at index 0
    assert_quantity_allclose(z[0], ref_geometry.a - ref_geometry.f)
    # Profile increases from vertex as r increases
    assert np.all(np.diff(z.value) >= 0)


# ---------------------------------------------------------------------------
# cassegrain_gain
# ---------------------------------------------------------------------------

def test_cassegrain_gain_units(ref_geometry: CassegrainGeometry) -> None:
    """Return value must be in dB."""
    gain = cassegrain_gain(
        ref_geometry, 2.2 * u.GHz, 1.0 * u.dimensionless_unscaled
    )
    assert gain.unit == u.dB(1)


def test_cassegrain_gain_reference_value(ref_geometry: CassegrainGeometry) -> None:
    """Gain must match the formula G = η·π²·(Dm²−Ds²)/λ² for the reference case."""
    freq = 2.2 * u.GHz
    eta = 1.0 * u.dimensionless_unscaled
    gain = cassegrain_gain(ref_geometry, freq, eta)

    # Independently derived expected value
    from astropy.constants import c

    wl = (c / freq).to(u.m)
    expected_linear = (
        eta
        * np.pi**2
        * (ref_geometry.Dm**2 - ref_geometry.Ds**2)
        / wl**2
    )
    expected_db = expected_linear.to(u.dB(1))
    assert_quantity_allclose(gain, expected_db, rtol=1e-6)
    # Physically reasonable for a 5 m dish at 2.2 GHz (η = 1)
    assert 40.0 < gain.value < 45.0


def test_cassegrain_gain_scales_with_efficiency(ref_geometry: CassegrainGeometry) -> None:
    """Halving efficiency reduces gain by ~3 dB."""
    freq = 2.2 * u.GHz
    g1 = cassegrain_gain(ref_geometry, freq, 1.0 * u.dimensionless_unscaled)
    g05 = cassegrain_gain(ref_geometry, freq, 0.5 * u.dimensionless_unscaled)
    assert g1.value - g05.value == pytest.approx(3.01, abs=0.01)


# ---------------------------------------------------------------------------
# Validity guards — design_min_blockage
# ---------------------------------------------------------------------------

def test_invalid_lm_nonpositive() -> None:
    """Lm <= 0 must raise ValueError."""
    with pytest.raises(ValueError, match="must be positive"):
        design_min_blockage(
            Dm=5 * u.m,
            F_over_D=0.317 * u.dimensionless_unscaled,
            Lm=-0.1 * u.m,
            Df=0.1 * u.m,
        )


def test_invalid_df_nonpositive() -> None:
    """Df <= 0 must raise ValueError."""
    with pytest.raises(ValueError, match="must be positive"):
        design_min_blockage(
            Dm=5 * u.m,
            F_over_D=0.317 * u.dimensionless_unscaled,
            Lm=1.1 * u.m,
            Df=0.0 * u.m,
        )


def test_invalid_lm_equal_to_f() -> None:
    """Lm = F gives f = 0, which must raise ValueError."""
    with pytest.raises(ValueError, match="must be positive"):
        design_min_blockage(
            Dm=5 * u.m,
            F_over_D=0.317 * u.dimensionless_unscaled,
            Lm=1.585 * u.m,  # Lm == F
            Df=0.1 * u.m,
        )


def test_invalid_lm_greater_than_f() -> None:
    """Lm > F gives f < 0, which must raise ValueError."""
    with pytest.raises(ValueError, match="must be positive"):
        design_min_blockage(
            Dm=5 * u.m,
            F_over_D=0.317 * u.dimensionless_unscaled,
            Lm=2.0 * u.m,  # Lm > F = 1.585
            Df=0.1 * u.m,
        )


def test_invalid_theta_e_denominator() -> None:
    """Inputs where Ds is large enough to make theta_e formula denominator non-positive."""
    with pytest.raises(ValueError, match="denominator"):
        design_min_blockage(
            Dm=5 * u.m,
            F_over_D=0.317 * u.dimensionless_unscaled,
            Lm=1.1 * u.m,
            Df=1.3 * u.m,  # forces Ds ≈ 4.25 m < Dm, but denominator ≈ -3 < 0
        )


def test_invalid_ds_ge_dm() -> None:
    """Large Df causes Ds >= Dm, which must raise ValueError."""
    with pytest.raises(ValueError, match="must be less than main reflector diameter"):
        design_min_blockage(
            Dm=5 * u.m,
            F_over_D=0.317 * u.dimensionless_unscaled,
            Lm=1.1 * u.m,
            Df=2.0 * u.m,  # forces Ds = F*Df/(2*f) >> Dm
        )


# ---------------------------------------------------------------------------
# Validity guards — cassegrain_gain
# ---------------------------------------------------------------------------

def test_cassegrain_gain_invalid_efficiency_zero() -> None:
    """efficiency = 0 must raise ValueError."""
    with pytest.raises(ValueError, match="out of range"):
        cassegrain_gain(
            design_min_blockage(
                Dm=5 * u.m,
                F_over_D=0.317 * u.dimensionless_unscaled,
                Lm=1.1 * u.m,
                Df=0.1 * u.m,
            ),
            2.2 * u.GHz,
            0.0 * u.dimensionless_unscaled,
        )


def test_cassegrain_gain_invalid_efficiency_over_one() -> None:
    """efficiency > 1 must raise ValueError."""
    with pytest.raises(ValueError, match="out of range"):
        cassegrain_gain(
            design_min_blockage(
                Dm=5 * u.m,
                F_over_D=0.317 * u.dimensionless_unscaled,
                Lm=1.1 * u.m,
                Df=0.1 * u.m,
            ),
            2.2 * u.GHz,
            1.5 * u.dimensionless_unscaled,
        )


def test_subreflector_profile_invalid_geometry() -> None:
    """f <= a in subreflector_profile must raise ValueError (Gregorian, not Cassegrain)."""
    with pytest.raises(ValueError, match="must be positive"):
        subreflector_profile(0 * u.m, a=0.5 * u.m, f=0.3 * u.m)  # f < a → invalid


def test_cassegrain_gain_invalid_geometry() -> None:
    """Manually constructed geometry with Ds >= Dm must raise ValueError."""
    bad_geom = CassegrainGeometry(
        Dm=1 * u.m,
        F=0.5 * u.m,
        Lm=0.4 * u.m,
        Ds=2 * u.m,  # Ds > Dm — invalid
        Ls=0.1 * u.m,
        a=0.05 * u.m,
        f=0.05 * u.m,
        theta_e=0.1 * u.rad,
    )
    with pytest.raises(ValueError, match="aperture area is non-positive"):
        cassegrain_gain(bad_geom, 10 * u.GHz, 0.65 * u.dimensionless_unscaled)


# ---------------------------------------------------------------------------
# Unit enforcement
# ---------------------------------------------------------------------------

def test_design_min_blockage_requires_quantities() -> None:
    """Plain floats (no units) must raise TypeError."""
    with pytest.raises(TypeError):
        design_min_blockage(5, 0.317, 1.1, 0.1)  # type: ignore[arg-type]


def test_main_reflector_profile_requires_quantity() -> None:
    with pytest.raises(TypeError):
        main_reflector_profile(1.0, 1.585)  # type: ignore[arg-type]


def test_subreflector_profile_requires_quantity() -> None:
    with pytest.raises(TypeError):
        subreflector_profile(0.1, 0.15, 0.24)  # type: ignore[arg-type]
