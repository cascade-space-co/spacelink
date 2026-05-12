import astropy.units as u
import numpy as np
import pytest
from astropy.tests.helper import assert_quantity_allclose

from spacelink.serialization.quantity import (
    QuantityModel,
    QuantityRangeModel,
    _resolve_unit,
)

# ---------------------------------------------------------------------------
# _resolve_unit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "unit_str, expected",
    [
        (None, u.dimensionless_unscaled),
        ("", u.dimensionless_unscaled),
        ("dimensionless", u.dimensionless_unscaled),
        ("dB", u.dB(1)),
        ("dBW", u.dB(u.W)),
        ("dBm", u.dB(u.mW)),
        ("dBHz", u.dB(u.Hz)),
        ("dBK", u.dB(u.K)),
        ("dB/K", u.dB(1 / u.K)),
        ("m", u.Unit("m")),
        ("Hz", u.Unit("Hz")),
        ("K / m", u.Unit("K / m")),
        ("dB(W)", u.Unit("dB(W)")),
        ("dB(mW)", u.Unit("dB(mW)")),
        ("dB(1/K)", u.Unit("dB(1/K)")),
    ],
)
def test_resolve_unit_valid(unit_str, expected):
    result = _resolve_unit(unit_str)
    assert result == expected


def test_resolve_unit_bad_raises():
    with pytest.raises(ValueError, match="Cannot resolve unit: 'badunit'"):
        _resolve_unit("badunit")


# ---------------------------------------------------------------------------
# from_astropy
# ---------------------------------------------------------------------------


def test_from_astropy_scalar_float():
    q = 1420.0 * u.MHz
    model = QuantityModel.from_astropy(q)
    assert isinstance(model.value, float)
    assert model.value == 1420.0
    assert model.unit == q.unit.to_string()


def test_from_astropy_numpy_array():
    q = np.array([1.0, 2.0, 3.0]) * u.m
    model = QuantityModel.from_astropy(q)
    assert isinstance(model.value, list)
    assert model.value == [1.0, 2.0, 3.0]


def test_from_astropy_zero_d_numpy_scalar():
    q = np.float64(5.0) * u.K
    model = QuantityModel.from_astropy(q)
    assert isinstance(model.value, float)


def test_from_astropy_non_quantity_raises():
    with pytest.raises(TypeError):
        QuantityModel.from_astropy(42.0)  # type: ignore[arg-type]


def test_from_astropy_non_quantity_string_raises():
    with pytest.raises(TypeError):
        QuantityModel.from_astropy("1420 MHz")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# to_astropy
# ---------------------------------------------------------------------------


def test_to_astropy_scalar_value_is_float():
    model = QuantityModel(value=3.7, unit="m")
    q = model.to_astropy()
    assert isinstance(q.value, float)


def test_to_astropy_list_value_is_ndarray():
    model = QuantityModel(value=[1.0, 2.0, 3.0], unit="m")
    q = model.to_astropy()
    assert isinstance(q.value, np.ndarray)


def test_to_astropy_empty_list_no_crash():
    model = QuantityModel(value=[], unit="m")
    q = model.to_astropy()
    assert isinstance(q.value, np.ndarray)
    assert len(q.value) == 0


def test_to_astropy_single_element_list_stays_array():
    model = QuantityModel(value=[1.0], unit="m")
    q = model.to_astropy()
    assert isinstance(q.value, np.ndarray)
    assert len(q.value) == 1


# ---------------------------------------------------------------------------
# Round-trips (from_astropy → to_astropy)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "original",
    [
        3.7 * u.dB(1),
        10.0 * u.dB(u.W),
        -10.0 * u.dB(u.mW),
        60.0 * u.dB(u.Hz),
        290.0 * u.dB(u.K),
        -25.0 * u.dB(1 / u.K),
        1420.0 * u.MHz,
        np.array([1.0, 2.0, 3.0]) * u.m,
    ],
)
def test_round_trip(original):
    model = QuantityModel.from_astropy(original)
    reconstructed = model.to_astropy()
    assert_quantity_allclose(reconstructed, original)


# ---------------------------------------------------------------------------
# Canonical dB forms go through u.Unit() (NOT in _LEGACY_ALIASES)
# ---------------------------------------------------------------------------


def test_resolve_unit_canonical_dbw_not_in_aliases():
    # "dB(W)" is canonical astropy form — must resolve via u.Unit(), not alias table
    result = _resolve_unit("dB(W)")
    assert result == u.Unit("dB(W)")


def test_resolve_unit_canonical_dbmw_not_in_aliases():
    result = _resolve_unit("dB(mW)")
    assert result == u.Unit("dB(mW)")


# ---------------------------------------------------------------------------
# JSON round-trips
# ---------------------------------------------------------------------------


def test_json_round_trip_scalar():
    original = QuantityModel(value=42.0, unit="Hz")
    dumped = original.model_dump()
    restored = QuantityModel.model_validate(dumped)
    assert restored == original


def test_json_round_trip_quantity_range_model():
    original = QuantityRangeModel(
        min=QuantityModel(value=1.0, unit="GHz"),
        max=QuantityModel(value=2.0, unit="GHz"),
    )
    dumped = original.model_dump()
    restored = QuantityRangeModel.model_validate(dumped)
    assert restored == original
