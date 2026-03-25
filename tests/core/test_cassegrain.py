"""Tests for the Cassegrain dual-reflector antenna design module."""

import astropy.units as u
import numpy as np
import pytest
from astropy.tests.helper import assert_quantity_allclose

from spacelink.core.cassegrain import (
    CassegrainGeometry,
    KildalEfficiency,
    cassegrain_gain,
    design_kildal_optimum,
    design_min_blockage,
    hyperboloid_focus_half_distance,
    hyperboloid_semi_axis,
    kildal_aperture_efficiency,
    kildal_blockage_parameter,
    kildal_blockage_term,
    kildal_diffraction_parameter,
    kildal_diffraction_term,
    kildal_feed_efficiency,
    kildal_interference_efficiency,
    kildal_optimal_ds_over_dm,
    main_reflector_profile,
    main_rim_angle,
    subreflector_diameter_min_blockage,
    subreflector_edge_angle,
    subreflector_feed_distance,
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


# Reference inputs (shared across primitive tests)
_Dm = 5 * u.m
_F = 1.585 * u.m
_Lm = 1.1 * u.m
_Df = 0.1 * u.m

# Reference outputs (Excel-verified)
_f_ref = 0.2425 * u.m
_Ds_ref = 0.326804124 * u.m
_theta_e_ref = 0.351304076 * u.rad
_Ls_ref = 0.395903483 * u.m
_a_ref = 0.153403483 * u.m


# ---------------------------------------------------------------------------
# design_min_blockage — reference case
# ---------------------------------------------------------------------------


def test_design_min_blockage_focal_length(ref_geometry: CassegrainGeometry) -> None:
    assert_quantity_allclose(ref_geometry.F, 1.585 * u.m, rtol=1e-4)


def test_design_min_blockage_f(ref_geometry: CassegrainGeometry) -> None:
    assert_quantity_allclose(ref_geometry.f, _f_ref, rtol=1e-4)


def test_design_min_blockage_Ds(ref_geometry: CassegrainGeometry) -> None:
    assert_quantity_allclose(ref_geometry.Ds, _Ds_ref, rtol=1e-6)


def test_design_min_blockage_theta_e(ref_geometry: CassegrainGeometry) -> None:
    assert_quantity_allclose(ref_geometry.theta_e, _theta_e_ref, rtol=1e-5)


def test_design_min_blockage_Ls(ref_geometry: CassegrainGeometry) -> None:
    assert_quantity_allclose(ref_geometry.Ls, _Ls_ref, rtol=1e-6)


def test_design_min_blockage_a(ref_geometry: CassegrainGeometry) -> None:
    assert_quantity_allclose(ref_geometry.a, _a_ref, rtol=1e-6)


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
# Granet geometry primitives
# ---------------------------------------------------------------------------


class TestHyperboloidFocusHalfDistance:
    def test_reference_value(self) -> None:
        """f = (F - Lm) / 2 for the reference case."""
        f = hyperboloid_focus_half_distance(_F, _Lm)
        assert_quantity_allclose(f, _f_ref, rtol=1e-6)

    def test_returns_quantity(self) -> None:
        f = hyperboloid_focus_half_distance(_F, _Lm)
        assert isinstance(f, u.Quantity)
        assert f.unit == u.m

    def test_invalid_lm_nonpositive(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            hyperboloid_focus_half_distance(_F, -0.1 * u.m)

    def test_invalid_lm_equals_F(self) -> None:
        """Lm = F gives f = 0."""
        with pytest.raises(ValueError, match="must be positive"):
            hyperboloid_focus_half_distance(_F, _F)

    def test_invalid_lm_greater_than_F(self) -> None:
        """Lm > F gives f < 0."""
        with pytest.raises(ValueError, match="must be positive"):
            hyperboloid_focus_half_distance(_F, 2.0 * u.m)

    def test_requires_quantity(self) -> None:
        with pytest.raises(TypeError):
            hyperboloid_focus_half_distance(1.585, 1.1)  # type: ignore[arg-type]


class TestSubreflectorDiameterMinBlockage:
    def test_reference_value(self) -> None:
        """Ds = F * Df / (2 * f) for the reference case."""
        Ds = subreflector_diameter_min_blockage(_F, _Df, _f_ref)
        assert_quantity_allclose(Ds, _Ds_ref, rtol=1e-6)

    def test_returns_quantity(self) -> None:
        Ds = subreflector_diameter_min_blockage(_F, _Df, _f_ref)
        assert isinstance(Ds, u.Quantity)
        assert Ds.unit == u.m

    def test_invalid_df_nonpositive(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            subreflector_diameter_min_blockage(_F, 0 * u.m, _f_ref)

    def test_requires_quantity(self) -> None:
        with pytest.raises(TypeError):
            subreflector_diameter_min_blockage(1.585, 0.1, 0.2425)  # type: ignore[arg-type]


class TestSubreflectorEdgeAngle:
    def test_reference_value(self) -> None:
        theta_e = subreflector_edge_angle(_F, _Dm, _Ds_ref, _f_ref)
        assert_quantity_allclose(theta_e, _theta_e_ref, rtol=1e-5)

    def test_returns_quantity_in_radians(self) -> None:
        theta_e = subreflector_edge_angle(_F, _Dm, _Ds_ref, _f_ref)
        assert isinstance(theta_e, u.Quantity)
        assert theta_e.unit == u.rad

    def test_positive_angle(self) -> None:
        theta_e = subreflector_edge_angle(_F, _Dm, _Ds_ref, _f_ref)
        assert theta_e > 0 * u.rad

    def test_invalid_ds_ge_dm(self) -> None:
        with pytest.raises(
            ValueError, match="must be less than main reflector diameter"
        ):
            subreflector_edge_angle(_F, _Dm, _Dm, _f_ref)

    def test_invalid_denominator(self) -> None:
        """Large Ds forces denominator negative."""
        with pytest.raises(ValueError, match="denominator"):
            subreflector_edge_angle(_F, _Dm, 4.25 * u.m, _f_ref)

    def test_requires_quantity(self) -> None:
        with pytest.raises(TypeError):
            subreflector_edge_angle(1.585, 5.0, 0.327, 0.2425)  # type: ignore[arg-type]


class TestSubreflectorFeedDistance:
    def test_reference_value(self) -> None:
        Ls = subreflector_feed_distance(_Dm, _f_ref, _F, _theta_e_ref)
        assert_quantity_allclose(Ls, _Ls_ref, rtol=1e-6)

    def test_returns_quantity(self) -> None:
        Ls = subreflector_feed_distance(_Dm, _f_ref, _F, _theta_e_ref)
        assert isinstance(Ls, u.Quantity)
        assert Ls.unit == u.m

    def test_requires_quantity(self) -> None:
        with pytest.raises(TypeError):
            subreflector_feed_distance(5.0, 0.2425, 1.585, 0.3513)  # type: ignore[arg-type]


class TestHyperboloidSemiAxis:
    def test_reference_value(self) -> None:
        a = hyperboloid_semi_axis(_Ls_ref, _f_ref)
        assert_quantity_allclose(a, _a_ref, rtol=1e-6)

    def test_returns_quantity(self) -> None:
        a = hyperboloid_semi_axis(_Ls_ref, _f_ref)
        assert isinstance(a, u.Quantity)
        assert a.unit == u.m

    def test_invalid_a_nonpositive(self) -> None:
        """Ls <= f gives a <= 0."""
        with pytest.raises(ValueError, match="must be positive"):
            hyperboloid_semi_axis(0.1 * u.m, 0.5 * u.m)

    def test_invalid_a_ge_f(self) -> None:
        """a >= f → invalid Cassegrain hyperboloid."""
        with pytest.raises(ValueError, match="must be less than f"):
            hyperboloid_semi_axis(1.5 * u.m, 0.5 * u.m)

    def test_requires_quantity(self) -> None:
        with pytest.raises(TypeError):
            hyperboloid_semi_axis(0.396, 0.2425)  # type: ignore[arg-type]


class TestMainRimAngle:
    def test_reference_value(self) -> None:
        """theta_0 = 2*arctan(Dm/(4F)) for reference case."""
        theta_0 = main_rim_angle(_Dm, _F)
        ratio = (_Dm / (4 * _F)).to(u.dimensionless_unscaled).value
        expected = (2 * np.arctan(ratio)) * u.rad
        assert_quantity_allclose(theta_0, expected, rtol=1e-10)

    def test_returns_quantity_in_radians(self) -> None:
        theta_0 = main_rim_angle(_Dm, _F)
        assert isinstance(theta_0, u.Quantity)
        assert theta_0.unit == u.rad

    def test_positive_angle(self) -> None:
        theta_0 = main_rim_angle(_Dm, _F)
        assert theta_0 > 0 * u.rad

    def test_angle_less_than_pi(self) -> None:
        theta_0 = main_rim_angle(_Dm, _F)
        assert theta_0 < np.pi * u.rad

    def test_requires_quantity(self) -> None:
        with pytest.raises(TypeError):
            main_rim_angle(5.0, 1.585)  # type: ignore[arg-type]


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
# Kildal efficiency primitives
# ---------------------------------------------------------------------------


# Use A0 = 0.5 for analytic reference values:
#   eta_f = 2*(0.5)^2 / ln(2) = 0.5 / 0.6931... = 0.72135...
#   Cb    = ln(2) / (1 - 0.5) = 0.6931 / 0.5    = 1.38629...
_A0_ref = 0.5 * u.dimensionless_unscaled
_eta_f_ref = 2 * 0.5**2 / np.log(2)  # ≈ 0.72135
_Cb_ref = np.log(2) / 0.5  # ≈ 1.38629


class TestKildalFeedEfficiency:
    def test_reference_value(self) -> None:
        eta_f = kildal_feed_efficiency(_A0_ref)
        assert_quantity_allclose(
            eta_f, _eta_f_ref * u.dimensionless_unscaled, rtol=1e-6
        )

    def test_returns_quantity(self) -> None:
        eta_f = kildal_feed_efficiency(_A0_ref)
        assert isinstance(eta_f, u.Quantity)
        assert eta_f.unit == u.dimensionless_unscaled

    def test_in_range(self) -> None:
        """Feed efficiency must be between 0 and 1 for any valid A0."""
        for a0 in [0.1, 0.3, 0.5, 0.7, 0.9]:
            eta_f = kildal_feed_efficiency(a0 * u.dimensionless_unscaled)
            assert 0 * u.dimensionless_unscaled < eta_f <= 1 * u.dimensionless_unscaled

    def test_invalid_a0_zero(self) -> None:
        with pytest.raises(ValueError, match="must satisfy 0 < A0 < 1"):
            kildal_feed_efficiency(0 * u.dimensionless_unscaled)

    def test_invalid_a0_one(self) -> None:
        with pytest.raises(ValueError, match="must satisfy 0 < A0 < 1"):
            kildal_feed_efficiency(1 * u.dimensionless_unscaled)

    def test_invalid_a0_negative(self) -> None:
        with pytest.raises(ValueError, match="must satisfy 0 < A0 < 1"):
            kildal_feed_efficiency(-0.1 * u.dimensionless_unscaled)

    def test_requires_quantity(self) -> None:
        with pytest.raises(TypeError):
            kildal_feed_efficiency(0.5)  # type: ignore[arg-type]


class TestKildalBlockageParameter:
    def test_reference_value(self) -> None:
        Cb = kildal_blockage_parameter(_A0_ref)
        assert_quantity_allclose(Cb, _Cb_ref * u.dimensionless_unscaled, rtol=1e-6)

    def test_returns_quantity(self) -> None:
        Cb = kildal_blockage_parameter(_A0_ref)
        assert isinstance(Cb, u.Quantity)
        assert Cb.unit == u.dimensionless_unscaled

    def test_positive(self) -> None:
        """Cb must be positive (it is the magnitude of a loss term)."""
        Cb = kildal_blockage_parameter(_A0_ref)
        assert Cb > 0 * u.dimensionless_unscaled

    def test_invalid_a0_zero(self) -> None:
        with pytest.raises(ValueError, match="must satisfy 0 < A0 < 1"):
            kildal_blockage_parameter(0 * u.dimensionless_unscaled)

    def test_invalid_a0_one(self) -> None:
        with pytest.raises(ValueError, match="must satisfy 0 < A0 < 1"):
            kildal_blockage_parameter(1 * u.dimensionless_unscaled)

    def test_requires_quantity(self) -> None:
        with pytest.raises(TypeError):
            kildal_blockage_parameter(0.5)  # type: ignore[arg-type]


class TestKildalDiffractionParameter:
    def test_returns_quantity(self) -> None:
        theta_0 = main_rim_angle(_Dm, _F)
        Cd = kildal_diffraction_parameter(theta_0, _theta_e_ref, _A0_ref)
        assert isinstance(Cd, u.Quantity)
        assert Cd.unit == u.dimensionless_unscaled

    def test_positive(self) -> None:
        theta_0 = main_rim_angle(_Dm, _F)
        Cd = kildal_diffraction_parameter(theta_0, _theta_e_ref, _A0_ref)
        assert Cd > 0 * u.dimensionless_unscaled

    def test_relation_to_Cb(self) -> None:
        """Cd ≈ Cb / π when theta_0 ≈ π/2 and theta_e → 0 (Kildal Eq. 29)."""
        # Use theta_0 = pi/2, theta_e = 0 → exact Eq. 29 applies
        theta_0 = (np.pi / 2) * u.rad
        theta_e_zero = 0 * u.rad
        Cb = kildal_blockage_parameter(_A0_ref)
        Cd = kildal_diffraction_parameter(theta_0, theta_e_zero, _A0_ref)
        # sin(pi/2) = 1, cos^2(0) = 1 → Cd = Cb/pi
        assert_quantity_allclose(Cd, Cb / np.pi, rtol=1e-6)

    def test_requires_quantity(self) -> None:
        with pytest.raises(TypeError):
            kildal_diffraction_parameter(1.33, 0.35, 0.5)  # type: ignore[arg-type]


class TestKildalBlockageTerm:
    def test_returns_quantity(self) -> None:
        Cb = kildal_blockage_parameter(_A0_ref)
        delta_cb = kildal_blockage_term(Cb, _Ds_ref, _Dm)
        assert isinstance(delta_cb, u.Quantity)
        assert delta_cb.unit == u.dimensionless_unscaled

    def test_negative(self) -> None:
        """Blockage term must be negative (loss)."""
        Cb = kildal_blockage_parameter(_A0_ref)
        delta_cb = kildal_blockage_term(Cb, _Ds_ref, _Dm)
        assert delta_cb < 0 * u.dimensionless_unscaled

    def test_reference_value(self) -> None:
        """delta_cb = -Cb * (Ds/Dm)^2 for reference case with A0=0.5."""
        Cb = kildal_blockage_parameter(_A0_ref)
        delta_cb = kildal_blockage_term(Cb, _Ds_ref, _Dm)
        d_over_D = (_Ds_ref / _Dm).to(u.dimensionless_unscaled)
        expected = -Cb * d_over_D**2
        assert_quantity_allclose(delta_cb, expected, rtol=1e-10)

    def test_requires_quantity(self) -> None:
        with pytest.raises(TypeError):
            kildal_blockage_term(1.386, 0.327, 5.0)  # type: ignore[arg-type]


class TestKildalDiffractionTerm:
    def test_returns_complex_quantity(self) -> None:
        theta_0 = main_rim_angle(_Dm, _F)
        Cd = kildal_diffraction_parameter(theta_0, _theta_e_ref, _A0_ref)
        delta_d = kildal_diffraction_term(Cd, _Ds_ref, _Dm, 2.2 * u.GHz, _A0_ref)
        assert isinstance(delta_d, u.Quantity)
        assert delta_d.unit == u.dimensionless_unscaled
        # Must be complex (non-zero imaginary part)
        assert delta_d.value.imag != 0

    def test_magnitude_small(self) -> None:
        """Diffraction term magnitude should be small (≪ 1) for a well-designed antenna."""
        theta_0 = main_rim_angle(_Dm, _F)
        Cd = kildal_diffraction_parameter(theta_0, _theta_e_ref, _A0_ref)
        delta_d = kildal_diffraction_term(Cd, _Ds_ref, _Dm, 2.2 * u.GHz, _A0_ref)
        assert abs(delta_d.value) < 0.5

    def test_scales_with_wavelength(self) -> None:
        """Diffraction term grows with wavelength (lower frequency → larger λ → larger Δ_d)."""
        theta_0 = main_rim_angle(_Dm, _F)
        Cd = kildal_diffraction_parameter(theta_0, _theta_e_ref, _A0_ref)
        delta_high = kildal_diffraction_term(Cd, _Ds_ref, _Dm, 10 * u.GHz, _A0_ref)
        delta_low = kildal_diffraction_term(Cd, _Ds_ref, _Dm, 1 * u.GHz, _A0_ref)
        assert abs(delta_low.value) > abs(delta_high.value)


class TestKildalInterferenceEfficiency:
    def test_returns_quantity(self) -> None:
        delta_cb = -0.05 * u.dimensionless_unscaled
        delta_d = 0 * u.dimensionless_unscaled
        eta_i = kildal_interference_efficiency(delta_cb, delta_d)
        assert isinstance(eta_i, u.Quantity)
        assert eta_i.unit == u.dimensionless_unscaled

    def test_zero_perturbation(self) -> None:
        """With no blockage or diffraction, eta_i = |1|^2 = 1."""
        delta_cb = 0 * u.dimensionless_unscaled
        delta_d = 0 * u.dimensionless_unscaled
        eta_i = kildal_interference_efficiency(delta_cb, delta_d)
        assert_quantity_allclose(eta_i, 1 * u.dimensionless_unscaled, rtol=1e-10)

    def test_real_loss_reduces_efficiency(self) -> None:
        """Pure real negative blockage reduces efficiency below 1."""
        delta_cb = -0.1 * u.dimensionless_unscaled
        delta_d = 0 * u.dimensionless_unscaled
        eta_i = kildal_interference_efficiency(delta_cb, delta_d)
        # |1 - 0.1|^2 = 0.81
        assert_quantity_allclose(eta_i, 0.81 * u.dimensionless_unscaled, rtol=1e-10)

    def test_complex_magnitude(self) -> None:
        """Verify |1 + Δ_cb + Δ_d|^2 with known complex input."""
        # 1 + (-0.1) + (0.1j) = 0.9 + 0.1j → |...|^2 = 0.81 + 0.01 = 0.82
        delta_cb = -0.1 * u.dimensionless_unscaled
        delta_d = 0.1j * u.dimensionless_unscaled
        eta_i = kildal_interference_efficiency(delta_cb, delta_d)
        assert_quantity_allclose(eta_i, 0.82 * u.dimensionless_unscaled, rtol=1e-10)


class TestKildalOptimalDsOverDm:
    def test_returns_dimensionless_quantity(self) -> None:
        theta_0 = main_rim_angle(_Dm, _F)
        ratio = kildal_optimal_ds_over_dm(
            theta_0, _theta_e_ref, _A0_ref, _Dm, 2.2 * u.GHz
        )
        assert isinstance(ratio, u.Quantity)
        assert ratio.unit == u.dimensionless_unscaled

    def test_ratio_in_valid_range(self) -> None:
        """Optimal Ds/Dm must be in (0, 1)."""
        theta_0 = main_rim_angle(_Dm, _F)
        ratio = kildal_optimal_ds_over_dm(
            theta_0, _theta_e_ref, _A0_ref, _Dm, 2.2 * u.GHz
        )
        assert 0 * u.dimensionless_unscaled < ratio < 1 * u.dimensionless_unscaled

    def test_increases_with_wavelength(self) -> None:
        """Larger wavelength → larger optimal subreflector (Eq. 31 ~ λ^(1/5))."""
        theta_0 = main_rim_angle(_Dm, _F)
        ratio_high_freq = kildal_optimal_ds_over_dm(
            theta_0, _theta_e_ref, _A0_ref, _Dm, 10 * u.GHz
        )
        ratio_low_freq = kildal_optimal_ds_over_dm(
            theta_0, _theta_e_ref, _A0_ref, _Dm, 1 * u.GHz
        )
        assert ratio_low_freq > ratio_high_freq

    def test_fifth_power_law(self) -> None:
        """Verify the λ^(1/5) scaling: doubling λ scales ratio by 2^(1/5)."""
        theta_0 = main_rim_angle(_Dm, _F)
        ratio_1 = kildal_optimal_ds_over_dm(
            theta_0, _theta_e_ref, _A0_ref, _Dm, 2.2 * u.GHz
        )
        ratio_half = kildal_optimal_ds_over_dm(
            theta_0,
            _theta_e_ref,
            _A0_ref,
            _Dm,
            4.4 * u.GHz,  # half the wavelength
        )
        # ratio ∝ λ^(1/5), so doubling f halves λ → ratio scales by (1/2)^(1/5)
        expected_scale = 2 ** (-1 / 5)
        assert_quantity_allclose(
            ratio_half / ratio_1, expected_scale * u.dimensionless_unscaled, rtol=1e-6
        )

    def test_small_angle_limit(self) -> None:
        """With theta_e = 0, cos^4(0) = 1, giving a purely frequency-dependent result."""
        theta_0 = main_rim_angle(_Dm, _F)
        theta_e_zero = 0 * u.rad
        ratio = kildal_optimal_ds_over_dm(
            theta_0, theta_e_zero, _A0_ref, _Dm, 2.2 * u.GHz
        )
        # Manually compute expected: [A0^2 / ((4pi)^2 * sin(theta_0)) * lambda/Dm]^(1/5)
        lam = (3e8 / 2.2e9) * u.m
        sin_t0 = np.sin(main_rim_angle(_Dm, _F).value)
        a0 = _A0_ref.value
        arg = (
            a0**2
            / ((4 * np.pi) ** 2 * sin_t0)
            * (lam / _Dm).to(u.dimensionless_unscaled).value
        )
        expected = arg**0.2 * u.dimensionless_unscaled
        assert_quantity_allclose(ratio, expected, rtol=1e-3)

    def test_requires_quantity(self) -> None:
        with pytest.raises(TypeError):
            kildal_optimal_ds_over_dm(1.33, 0.35, 0.5, 5.0, 2.2e9)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# design_kildal_optimum
# ---------------------------------------------------------------------------


class TestDesignKildalOptimum:
    def test_returns_valid_geometry(self) -> None:
        geom = design_kildal_optimum(
            Dm=_Dm,
            F_over_D=0.317 * u.dimensionless_unscaled,
            Lm=_Lm,
            frequency=2.2 * u.GHz,
            A0=_A0_ref,
        )
        assert isinstance(geom, CassegrainGeometry)

    def test_eccentricity_gt_one(self) -> None:
        """Result must be a valid Cassegrain hyperboloid (e > 1)."""
        geom = design_kildal_optimum(
            Dm=_Dm,
            F_over_D=0.317 * u.dimensionless_unscaled,
            Lm=_Lm,
            frequency=2.2 * u.GHz,
            A0=_A0_ref,
        )
        assert geom.eccentricity > 1 * u.dimensionless_unscaled

    def test_ds_less_than_dm(self) -> None:
        geom = design_kildal_optimum(
            Dm=_Dm,
            F_over_D=0.317 * u.dimensionless_unscaled,
            Lm=_Lm,
            frequency=2.2 * u.GHz,
            A0=_A0_ref,
        )
        assert geom.Ds < geom.Dm

    def test_focal_length_consistent(self) -> None:
        """F = F_over_D * Dm regardless of strategy."""
        geom = design_kildal_optimum(
            Dm=_Dm,
            F_over_D=0.317 * u.dimensionless_unscaled,
            Lm=_Lm,
            frequency=2.2 * u.GHz,
            A0=_A0_ref,
        )
        assert_quantity_allclose(geom.F, 0.317 * _Dm, rtol=1e-6)

    def test_larger_dish_smaller_relative_ds(self) -> None:
        """For fixed D/λ ratio, result should scale with Dm."""
        geom_small = design_kildal_optimum(
            Dm=1 * u.m,
            F_over_D=0.317 * u.dimensionless_unscaled,
            Lm=0.22 * u.m,
            frequency=2.2 * u.GHz,
            A0=_A0_ref,
        )
        geom_large = design_kildal_optimum(
            Dm=10 * u.m,
            F_over_D=0.317 * u.dimensionless_unscaled,
            Lm=2.2 * u.m,
            frequency=2.2 * u.GHz,
            A0=_A0_ref,
        )
        ratio_small = (geom_small.Ds / geom_small.Dm).to(u.dimensionless_unscaled)
        ratio_large = (geom_large.Ds / geom_large.Dm).to(u.dimensionless_unscaled)
        # Larger antenna has larger D/λ → smaller optimal Ds/Dm
        assert ratio_large < ratio_small

    def test_invalid_a0(self) -> None:
        with pytest.raises(ValueError, match="must satisfy 0 < A0 < 1"):
            design_kildal_optimum(
                Dm=_Dm,
                F_over_D=0.317 * u.dimensionless_unscaled,
                Lm=_Lm,
                frequency=2.2 * u.GHz,
                A0=1 * u.dimensionless_unscaled,
            )

    def test_requires_quantity(self) -> None:
        with pytest.raises(TypeError):
            design_kildal_optimum(5.0, 0.317, 1.1, 2.2e9, 0.5)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# kildal_aperture_efficiency
# ---------------------------------------------------------------------------


class TestKildalApertureEfficiency:
    def test_returns_kildal_efficiency(self, ref_geometry: CassegrainGeometry) -> None:
        result = kildal_aperture_efficiency(ref_geometry, 2.2 * u.GHz, _A0_ref)
        assert isinstance(result, KildalEfficiency)

    def test_all_fields_are_quantities(self, ref_geometry: CassegrainGeometry) -> None:
        result = kildal_aperture_efficiency(ref_geometry, 2.2 * u.GHz, _A0_ref)
        for field_name in ["eta_f", "delta_cb", "eta_i", "eta_a"]:
            val = getattr(result, field_name)
            assert isinstance(val, u.Quantity), f"{field_name} is not a Quantity"
        assert isinstance(result.delta_d, u.Quantity)

    def test_eta_f_in_range(self, ref_geometry: CassegrainGeometry) -> None:
        result = kildal_aperture_efficiency(ref_geometry, 2.2 * u.GHz, _A0_ref)
        assert (
            0 * u.dimensionless_unscaled < result.eta_f <= 1 * u.dimensionless_unscaled
        )

    def test_delta_cb_negative(self, ref_geometry: CassegrainGeometry) -> None:
        """Centre-blockage term must be negative."""
        result = kildal_aperture_efficiency(ref_geometry, 2.2 * u.GHz, _A0_ref)
        assert result.delta_cb.real < 0 * u.dimensionless_unscaled

    def test_delta_d_complex(self, ref_geometry: CassegrainGeometry) -> None:
        """Diffraction term must have non-zero imaginary part."""
        result = kildal_aperture_efficiency(ref_geometry, 2.2 * u.GHz, _A0_ref)
        assert result.delta_d.value.imag != 0

    def test_eta_a_equals_product(self, ref_geometry: CassegrainGeometry) -> None:
        """η_a = η_f · η_i."""
        result = kildal_aperture_efficiency(ref_geometry, 2.2 * u.GHz, _A0_ref)
        assert_quantity_allclose(result.eta_a, result.eta_f * result.eta_i, rtol=1e-10)

    def test_eta_a_less_than_one(self, ref_geometry: CassegrainGeometry) -> None:
        """Real antenna efficiency must be below 1."""
        result = kildal_aperture_efficiency(ref_geometry, 2.2 * u.GHz, _A0_ref)
        assert result.eta_a < 1 * u.dimensionless_unscaled

    def test_eta_a_physically_reasonable(
        self, ref_geometry: CassegrainGeometry
    ) -> None:
        """Efficiency > 0.5 for a well-designed 5 m dish at 2.2 GHz with moderate taper."""
        result = kildal_aperture_efficiency(ref_geometry, 2.2 * u.GHz, _A0_ref)
        assert result.eta_a > 0.5 * u.dimensionless_unscaled


# ---------------------------------------------------------------------------
# cassegrain_gain
# ---------------------------------------------------------------------------


def test_cassegrain_gain_units(ref_geometry: CassegrainGeometry) -> None:
    """Return value must be in dB."""
    gain = cassegrain_gain(ref_geometry, 2.2 * u.GHz, 1.0 * u.dimensionless_unscaled)
    assert gain.unit == u.dB(1)


def test_cassegrain_gain_reference_value(ref_geometry: CassegrainGeometry) -> None:
    """Gain must match the formula G = η·π²·(Dm²−Ds²)/λ² for the reference case."""
    freq = 2.2 * u.GHz
    eta = 1.0 * u.dimensionless_unscaled
    gain = cassegrain_gain(ref_geometry, freq, eta)

    # Independently derived expected value
    from astropy.constants import c

    wl = (c / freq).to(u.m)
    expected_linear = eta * np.pi**2 * (ref_geometry.Dm**2 - ref_geometry.Ds**2) / wl**2
    expected_db = expected_linear.to(u.dB(1))
    assert_quantity_allclose(gain, expected_db, rtol=1e-6)
    # Physically reasonable for a 5 m dish at 2.2 GHz (η = 1)
    assert 40.0 < gain.value < 45.0


def test_cassegrain_gain_scales_with_efficiency(
    ref_geometry: CassegrainGeometry,
) -> None:
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
