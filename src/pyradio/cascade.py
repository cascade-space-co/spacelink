"""
Cascade module for cascaded RF stage analysis.

This module provides classes to model individual RF stages (e.g., amplifiers,
attenuators, filters) and to compute cascaded performance metrics such as
overall gain, noise figure, noise temperature, and input-referred P1dB.
"""

import math
from typing import List, Optional, Dict, Any
from pint import Quantity
from pyradio.noise import (
    noise_figure_to_temperature,
    temperature_to_noise_figure,
    cascaded_noise_figure,
    cascaded_noise_temperature,
)
from pyradio.units import (
    Q_,
    dB,
    dBm,
    mW,
    K,
    dimensionless,
    vswr_to_return_loss,
    return_loss_to_vswr,
    mismatch_loss,
)


class Stage:
    """
    Represents a single stage in an RF cascade.

    Each stage has parameters such as gain or loss, noise figure or noise
    temperature, P1dB, and input/output return loss or VSWR.

    Args:
        label: Descriptive label for the stage.
        gain: Gain in decibels. Mutually exclusive with loss.
        loss: Loss in decibels (positive value). Mutually exclusive with gain.
        noise_figure: Noise figure in decibels. Mutually exclusive with noise_temp.
        noise_temp: Noise temperature in Kelvin. Mutually exclusive with noise_figure.
        p1db_dbm: Output 1dB compression point in dBm.
        input_rl_db: Input return loss in dB. Mutually exclusive with input_vswr.
        output_rl_db: Output return loss in dB. Mutually exclusive with output_vswr.
        input_vswr: Input VSWR (>=1). Mutually exclusive with input_rl_db.
        output_vswr: Output VSWR (>=1). Mutually exclusive with output_rl_db.

    Raises:
        ValueError: If invalid parameter combinations or values are provided.
    """

    def __init__(
        self,
        label: str,
        gain: Optional[Quantity] = None,
        loss: Optional[Quantity] = None,
        noise_figure: Optional[Quantity] = None,
        noise_temp: Optional[Quantity] = None,
        p1db_dbm: Optional[Quantity] = None,
        input_rl_db: Optional[Quantity] = None,
        output_rl_db: Optional[Quantity] = None,
        input_vswr: Optional[Quantity] = None,
        output_vswr: Optional[Quantity] = None,
    ) -> None:
        # Label
        self.label: str = label

        # --- Gain/Loss (store as Quantity in dB) ---
        if gain is not None and loss is not None:
            raise ValueError(f"Stage '{label}': Cannot specify both gain and loss.")
        if loss is not None:
            # Loss in dB is positive; stored as negative gain
            loss_q = (
                loss.to(dB) if isinstance(loss, Quantity) else Q_(loss, dB)
            )
            self._gain: Quantity = -loss_q
        elif gain is not None:
            gain_q = gain.to(dB) if isinstance(gain, Quantity) else Q_(gain, dB)
            self._gain = gain_q
        else:
            self._gain = Q_(0.0, dB)

        # --- Noise Temperature (internal) ---
        # Accept either noise_figure or noise_temp, store only noise_temp
        # Preserve original noise figure for to_dict if provided
        self._orig_noise_figure: Optional[Quantity] = None
        if noise_figure is not None and noise_temp is not None:
            nf_q = (
                noise_figure.to(dB)
                if isinstance(noise_figure, Quantity)
                else Q_(noise_figure, dB)
            )
            nt_q = (
                noise_temp.to(K)
                if isinstance(noise_temp, Quantity)
                else Q_(noise_temp, K)
            )
            calc_nt = noise_figure_to_temperature(nf_q)
            if not math.isclose(calc_nt.magnitude, nt_q.magnitude, rel_tol=1e-6):
                raise ValueError(
                    f"Stage '{label}': noise_figure ({nf_q}) and noise_temp ({nt_q}) "
                    "are inconsistent."
                )
            self._noise_temp = nt_q
            self._orig_noise_figure = nf_q
        elif noise_figure is not None:
            nf_q = (
                noise_figure.to(dB)
                if isinstance(noise_figure, Quantity)
                else Q_(noise_figure, dB)
            )
            self._noise_temp = noise_figure_to_temperature(nf_q)
            self._orig_noise_figure = nf_q
        elif noise_temp is not None:
            nt_q = (
                noise_temp.to(K)
                if isinstance(noise_temp, Quantity)
                else Q_(noise_temp, K)
            )
            if nt_q.magnitude < 0:
                raise ValueError(
                    f"Stage '{label}': noise_temp cannot be negative ({nt_q})."
                )
            self._noise_temp = nt_q
        else:
            self._noise_temp = None

        # --- P1dB (store as Quantity in dBm) ---
        if p1db_dbm is not None:
            self._p1db_dbm: Quantity = (
                p1db_dbm.to(dBm)
                if isinstance(p1db_dbm, Quantity)
                else Q_(p1db_dbm, dBm)
            )
        else:
            self._p1db_dbm = None  # type: ignore

        # --- Return Loss / VSWR ---
        # Input side: convert inputs to raw floats
        in_rl_val = (
            input_rl_db.to(dB).magnitude
            if isinstance(input_rl_db, Quantity)
            else input_rl_db
        )
        in_vswr_val = (
            input_vswr.to(dimensionless).magnitude
            if isinstance(input_vswr, Quantity)
            else input_vswr
        )
        raw_in_rl, raw_in_vswr = self._process_rl_vswr(
            in_rl_val, in_vswr_val, side="input"
        )
        self._input_rl_db = Q_(raw_in_rl, dB) if raw_in_rl is not None else None
        self._input_vswr = (
            Q_(raw_in_vswr, dimensionless) if raw_in_vswr is not None else None
        )
        # Output side: convert inputs to raw floats
        out_rl_val = (
            output_rl_db.to(dB).magnitude
            if isinstance(output_rl_db, Quantity)
            else output_rl_db
        )
        out_vswr_val = (
            output_vswr.to(dimensionless).magnitude
            if isinstance(output_vswr, Quantity)
            else output_vswr
        )
        raw_out_rl, raw_out_vswr = self._process_rl_vswr(
            out_rl_val, out_vswr_val, side="output"
        )
        self._output_rl_db = Q_(raw_out_rl, dB) if raw_out_rl is not None else None
        self._output_vswr = (
            Q_(raw_out_vswr, dimensionless) if raw_out_vswr is not None else None
        )

    def _process_rl_vswr(
        self,
        rl_db: Optional[float],
        vswr: Optional[float],
        side: str = "input",
    ) -> (Optional[float], Optional[float]):
        """
        Process return loss and VSWR parameters for one side.

        Returns a tuple of (rl_db, vswr).
        """
        out_rl: Optional[float] = None
        out_vswr: Optional[float] = None
        # Both provided: check consistency
        if rl_db is not None and vswr is not None:
            if rl_db < 0:
                raise ValueError(
                    f"Stage '{self.label}': {side} return loss cannot be negative ({rl_db})."
                )
            if vswr < 1.0:
                raise ValueError(
                    f"Stage '{self.label}': {side} VSWR must be >= 1 ({vswr})."
                )
            calc_vswr = return_loss_to_vswr(rl_db)
            if not math.isclose(vswr, calc_vswr, rel_tol=1e-4):
                raise ValueError(
                    f"Stage '{self.label}': {side} rl_db ({rl_db}) and vswr ({vswr}) "
                    f"are inconsistent (calc {calc_vswr})."
                )
            out_rl = float(rl_db)
            out_vswr = float(vswr)
        elif rl_db is not None:
            if rl_db < 0:
                raise ValueError(
                    f"Stage '{self.label}': {side} return loss cannot be negative ({rl_db})."
                )
            out_rl = float(rl_db)
            out_vswr = return_loss_to_vswr(out_rl)
        elif vswr is not None:
            if vswr < 1.0:
                raise ValueError(
                    f"Stage '{self.label}': {side} VSWR must be >= 1 ({vswr})."
                )
            out_vswr = float(vswr)
            out_rl = vswr_to_return_loss(out_vswr)
        return out_rl, out_vswr

    # --- Properties ---
    @property
    def gain(self) -> Quantity:
        """Gain in decibels (Quantity with dB unit)."""
        return self._gain

    @property
    def gain_lin(self) -> Quantity:
        """Linear gain (ratio, dimensionless)."""
        lin_q = self._gain.to(dimensionless)
        mag = float(lin_q.magnitude)
        # Round to integer if very close to avoid floating-point artifacts
        if abs(mag - round(mag)) < 1e-9:
            mag = round(mag)
        return Q_(mag, dimensionless)

    @property
    def noise_figure(self) -> Optional[Quantity]:
        """Noise figure (Quantity with dB unit)."""
        if self._noise_temp is None:
            return None
        # If original NF was provided, return it to preserve exact value
        if getattr(self, "_orig_noise_figure", None) is not None:
            return self._orig_noise_figure
        return temperature_to_noise_figure(self._noise_temp)

    @property
    def noise_temp(self) -> Optional[Quantity]:
        """Noise temperature (Quantity with K unit)."""
        return self._noise_temp

    @property
    def noise_factor(self) -> Optional[Quantity]:
        """Noise factor (linear, dimensionless)."""
        nf = self.noise_figure
        return nf.to(dimensionless) if nf is not None else None

    @property
    def p1db_dbm(self) -> Optional[Quantity]:
        """Output P1dB (Quantity with dBm unit)."""
        return self._p1db_dbm

    @property
    def p1db_mw(self) -> Optional[Quantity]:
        """Output P1dB (Quantity with mW unit)."""
        return self._p1db_dbm.to(mW) if self._p1db_dbm is not None else None

    @property
    def input_rl_db(self) -> Optional[Quantity]:
        """Input return loss (Quantity with dB unit)."""
        # Return raw return loss in dB as float
        return (
            float(self._input_rl_db.magnitude)
            if self._input_rl_db is not None
            else None
        )

    @property
    def output_rl_db(self) -> Optional[Quantity]:
        """Output return loss (Quantity with dB unit)."""
        # Return raw return loss in dB as float
        return (
            float(self._output_rl_db.magnitude)
            if self._output_rl_db is not None
            else None
        )

    @property
    def input_vswr(self) -> Optional[Quantity]:
        """Input VSWR (Quantity dimensionless)."""
        return self._input_vswr

    @property
    def output_vswr(self) -> Optional[Quantity]:
        """Output VSWR (Quantity dimensionless)."""
        return self._output_vswr

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize stage to a dictionary for YAML/JSON.
        """
        data: Dict[str, Any] = {"label": self.label}
        # Gain or loss (store magnitudes as floats)
        if self._gain.magnitude >= 0:
            data["gain"] = float(self._gain.magnitude)
        else:
            # loss is stored as negative gain
            data["loss"] = float((-self._gain).magnitude)
        # Noise figure
        if self.noise_figure is not None:
            data["noise_figure"] = float(self.noise_figure.magnitude)
        # P1dB
        if self._p1db_dbm is not None:
            data["p1db_dbm"] = float(self._p1db_dbm.magnitude)
        # Return loss
        if self._input_rl_db is not None:
            data["input_rl_db"] = float(self._input_rl_db.magnitude)
        if self._output_rl_db is not None:
            data["output_rl_db"] = float(self._output_rl_db.magnitude)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Stage":
        """
        Create a Stage from a dictionary.
        """
        return cls(**data)  # type: ignore

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Stage):
            return False
        return self.to_dict() == other.to_dict()

    def __repr__(self) -> str:
        return f"Stage({self.to_dict()})"


class Cascade:
    """
    Represents a cascade of RF stages and computes overall metrics.

    Args:
        stages: Optional list of Stage objects to initialize the cascade.

    Raises:
        TypeError: If any item in stages is not a Stage.
    """

    def __init__(self, stages: Optional[List[Stage]] = None) -> None:
        self.stages: List[Stage] = []
        if stages is not None:
            if not all(isinstance(s, Stage) for s in stages):
                raise TypeError("All items in stages must be Stage objects.")
            self.stages = list(stages)

    def add_stage(self, stage: Stage) -> None:
        """Add a Stage to the end of the cascade."""
        if not isinstance(stage, Stage):
            raise TypeError("Can only add Stage objects to the cascade.")
        self.stages.append(stage)

    def cascaded_gain(self) -> Quantity:
        """Total cascaded gain in decibels."""
        """Total cascaded gain in decibels, accounting for mismatch losses."""
        # No stages: zero gain
        if not self.stages:
            return Q_(0.0, dB)
        # Sum stage gains (in dB)
        total_db_mag = sum(stage.gain.to(dB).magnitude for stage in self.stages)
        # Subtract mismatch losses between stages: for each stage after the first,
        # apply mismatch_loss using the stage input return loss (if specified).
        mismatch_db = 0.0
        for stage in self.stages[1:]:
            rl = stage.input_rl_db
            if rl is not None:
                # rl is a float in dB
                mismatch_db += mismatch_loss(rl)
        total_db_mag -= mismatch_db
        return Q_(total_db_mag, dB)

    def cascaded_noise_figure_db(self) -> Quantity:
        """
        Total cascaded noise figure in decibels using Friis formula.

        Raises:
            ValueError: If chain is empty or any stage lacks noise/gain.
        """
        if not self.stages:
            raise ValueError("Cannot calculate noise figure for empty cascade.")
        noise_factors = []
        gains_lin = []
        for i, stage in enumerate(self.stages):
            nf = stage.noise_factor
            gl = stage.gain_lin
            if nf is None:
                raise ValueError(f"Stage {i} ('{stage.label}') missing noise figure.")
            if gl is None:
                raise ValueError(f"Stage {i} ('{stage.label}') missing gain.")
            noise_factors.append(nf)
            gains_lin.append(gl)
        return cascaded_noise_figure(noise_factors, gains_lin)

    def cascaded_noise_temperature_k(self) -> Quantity:
        """Total cascaded noise temperature."""
        if not self.stages:
            raise ValueError("Cannot calculate noise temperature for empty cascade.")
        noise_temps = []
        gains_lin = []
        for i, stage in enumerate(self.stages):
            nt = stage.noise_temp
            gl = stage.gain_lin
            if nt is None:
                raise ValueError(
                    f"Stage {i} ('{stage.label}') missing noise temperature."
                )
            if gl is None:
                raise ValueError(f"Stage {i} ('{stage.label}') missing gain.")
            noise_temps.append(nt)
            gains_lin.append(gl)
        return cascaded_noise_temperature(noise_temps, gains_lin)

    def input_referred_p1db_dbm(self) -> Optional[Quantity]:
        """
        Input-referred 1dB compression point in dBm.

        Raises:
            ValueError: If chain is empty or any stage lacks P1dB/gain.
        """
        if not self.stages:
            raise ValueError("Cannot calculate P1dB for empty cascade.")
        # Sum reciprocal of output P1dB powers referred to chain output
        total_recip: float = 0.0
        gain_after_val: float = 1.0
        for stage in reversed(self.stages):
            p1_q = stage.p1db_mw
            if p1_q is None:
                raise ValueError(f"Stage '{stage.label}' missing P1dB.")
            # Work in milliwatts
            p1_mw = p1_q.to(mW).magnitude
            total_recip += 1.0 / (p1_mw * gain_after_val)
            # Accumulate linear gain
            gain_lin_q = stage.gain_lin
            gain_after_val *= gain_lin_q.to(dimensionless).magnitude  # type: ignore
        if total_recip <= 0:
            return None
        out_p1_mw: float = 1.0 / total_recip
        # Convert to dBm quantity
        out_p1_dbm_q: Quantity = Q_(out_p1_mw, mW).to(dBm)
        # Refer back to input (subtract total gain in dB)
        total_gain = self.cascaded_gain().to(dB).magnitude
        input_p1_dbm_mag: float = out_p1_dbm_q.to(dBm).magnitude - total_gain
        return Q_(input_p1_dbm_mag, dBm)

    def __len__(self) -> int:
        return len(self.stages)

    def __getitem__(self, idx: int) -> Stage:
        return self.stages[idx]

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize cascade to a dictionary for YAML/JSON.
        """
        return {"stages": [stage.to_dict() for stage in self.stages]}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Cascade":
        """
        Create a Cascade from a dictionary.
        """
        stages_data = data.get("stages", []) or []
        stages = [Stage.from_dict(d) for d in stages_data]
        return cls(stages)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Cascade):
            return False
        return self.to_dict() == other.to_dict()

    def __repr__(self) -> str:
        return f"Cascade({self.to_dict()})"
