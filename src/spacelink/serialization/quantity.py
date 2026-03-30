import astropy.units as u
import numpy as np
from astropy.units import Quantity
from pydantic import BaseModel, ConfigDict, Field

_LEGACY_ALIASES: dict[str, u.UnitBase] = {
    "dB": u.dB(1),  # bare "dB" → dB(1), i.e. dB relative to 1 (dimensionless ratio)
    "dBW": u.dB(u.W),
    "dBm": u.dB(u.mW),
    "dBHz": u.dB(u.Hz),
    "dBK": u.dB(u.K),
    "dB/K": u.dB(1 / u.K),
}


def _resolve_unit(unit_str: str | None) -> u.UnitBase:
    """Return the astropy unit object for a unit string.

    Falls back to u.Unit() for standard and composite units (e.g. "K / m",
    "W / Hz", "dB(W)"). _LEGACY_ALIASES handles the dB short-forms that
    cascade-designer stores in JSON and that u.Unit() cannot parse.
    """
    if not unit_str:
        return u.dimensionless_unscaled
    unit_str = unit_str.strip()
    if not unit_str or unit_str == "dimensionless":
        return u.dimensionless_unscaled
    if unit_str in _LEGACY_ALIASES:
        return _LEGACY_ALIASES[unit_str]
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
