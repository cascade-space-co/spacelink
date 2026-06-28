import math
import re

import astropy.units as u
import numpy as np
from astropy.units import Quantity
from pydantic import BaseModel, ConfigDict, Field, field_validator

# Common dB short-forms from RF engineering notation that astropy's parser
# does not accept (or parses wrongly). "dB/K" must be aliased: u.Unit() parses
# it as a *linear* dB÷K composite, not the *logarithmic* dB(1/K) that G/T
# requires. _resolve_unit normalizes whitespace, so "dB / K" maps here too.
# Canonical forms ("dB(W)", "dB(1 / K)", etc.) resolve via u.Unit() directly.
_DB_SHORT_FORMS: dict[str, u.UnitBase] = {
    "dB": u.dB(1),  # intentionally resolved as dB(1), the dimensionless dB ratio
    "dBW": u.dB(u.W),
    "dBm": u.dB(u.mW),
    "dBHz": u.dB(u.Hz),
    "dBK": u.dB(u.K),
    "dB/K": u.dB(1 / u.K),
}


def _resolve_unit(unit_str: str | None) -> u.UnitBase:
    """Return the astropy unit object for a unit string.

    Falls back to u.Unit() for standard and composite units (e.g. "K / m",
    "W / Hz", "dB(W)"). _DB_SHORT_FORMS handles the dB short-forms that
    u.Unit() cannot parse (or parses incorrectly, e.g. "dB/K").
    """
    if not unit_str:
        return u.dimensionless_unscaled
    unit_str = unit_str.strip()
    if not unit_str or unit_str == "dimensionless":
        return u.dimensionless_unscaled
    # Normalize whitespace around the slash for the alias lookup only, so that
    # "dB / K" matches the "dB/K" short-form instead of falling through to
    # u.Unit() (which would parse it as a linear composite).
    alias_key = re.sub(r"\s*/\s*", "/", unit_str)
    if alias_key in _DB_SHORT_FORMS:
        return _DB_SHORT_FORMS[alias_key]
    try:
        return u.Unit(unit_str)
    except ValueError:
        raise ValueError(f"Cannot resolve unit: {unit_str!r}") from None


class QuantityModel(BaseModel):
    value: float | list[float] = Field(
        ..., description="Numeric value (scalar or array)"
    )
    unit: str | None = Field(None, description="Unit string (None → dimensionless)")

    model_config = ConfigDict(extra="forbid")

    @field_validator("value")
    @classmethod
    def _reject_non_finite(cls, value: float | list[float]) -> float | list[float]:
        values = value if isinstance(value, list) else [value]
        if not all(math.isfinite(v) for v in values):
            raise ValueError("value must be finite (no NaN or inf)")
        return value

    @classmethod
    def from_astropy(cls, q: Quantity) -> "QuantityModel":
        if not isinstance(q, Quantity):
            raise TypeError(f"Expected astropy Quantity, got {type(q).__name__}")
        val = q.value
        if isinstance(val, np.ndarray) and val.shape != ():
            value: float | list[float] = val.tolist()
        else:
            value = float(val)
        return cls(value=value, unit=q.unit.to_string())

    def to_astropy(self) -> Quantity:
        unit = _resolve_unit(self.unit)
        if isinstance(self.value, list):
            return np.array(self.value) * unit  # array path
        return self.value * unit  # scalar path (avoids 0-d ndarray)


class QuantityRangeModel(BaseModel):
    """Represents a range of values with units (e.g., frequency range).

    No min <= max or unit compatibility validation — callers handle at use time.
    """

    min: QuantityModel = Field(..., description="Minimum value")
    max: QuantityModel = Field(..., description="Maximum value")

    model_config = ConfigDict(extra="forbid")
