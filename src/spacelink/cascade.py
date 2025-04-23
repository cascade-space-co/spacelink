"""
Cascade module for cascaded RF stage analysis.

This module provides classes to model individual RF stages (e.g., amplifiers,
attenuators, filters) and to compute cascaded performance metrics such as
overall gain, noise figure, noise temperature, and input-referred P1dB.
"""

import math
import numpy as np
from typing import List, Optional, Dict, Any
import astropy.units as u
from astropy.units import Quantity

from spacelink.noise import (
    noise_figure_to_temperature,
    temperature_to_noise_figure,
    cascaded_noise_figure,
    cascaded_noise_temperature,
)
from spacelink.units import (
    vswr_to_return_loss,
    return_loss_to_vswr,
    mismatch_loss,
    to_linear,
    to_dB,
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
            loss_q = loss if isinstance(loss, Quantity) else loss * u.dB
            self._gain: Quantity = -loss_q
        elif gain is not None:
            gain_q = gain if isinstance(gain, Quantity) else gain * u.dB
            self._gain = gain_q
        else:
            self._gain = 0.0 * u.dB

        # --- Noise Temperature (internal) ---
        # Accept either noise_figure or noise_temp, store only noise_temp
        # Preserve original noise figure for to_dict if provided
        self._orig_noise_figure: Optional[Quantity] = None
        if noise_figure is not None and noise_temp is not None:
            nf_q = noise_figure if isinstance(noise_figure, Quantity) else noise_figure * u.dB
            nt_q = noise_temp if isinstance(noise_temp, Quantity) else noise_temp * u.K
            calc_nt = noise_figure_to_temperature(nf_q)
            if not math.isclose(calc_nt.value, nt_q.value, rel_tol=1e-6):
                raise ValueError(
                    f"Stage '{label}': noise_figure ({nf_q}) and "
                    f"noise_temp ({nt_q}) are inconsistent."
                )
            self._noise_temp = nt_q
            self._orig_noise_figure = nf_q
        elif noise_figure is not None:
            nf_q = noise_figure if isinstance(noise_figure, Quantity) else noise_figure * u.dB
            self._noise_temp = noise_figure_to_temperature(nf_q)
            self._orig_noise_figure = nf_q
        elif noise_temp is not None:
            nt_q = noise_temp if isinstance(noise_temp, Quantity) else noise_temp * u.K
            if nt_q < 0 * u.K:
                raise ValueError(
                    f"Stage '{label}': noise_temp cannot be negative ({nt_q})."
                )
            self._noise_temp = nt_q
        else:
            self._noise_temp = None

        # --- P1dB (store as Quantity in dBm) ---
        if p1db_dbm is not None:
            self._p1db_dbm: Quantity = p1db_dbm if isinstance(p1db_dbm, Quantity) else p1db_dbm * u.dBm
        else:
            self._p1db_dbm = None  # type: ignore

        # --- Return Loss / VSWR ---
        # Input side
        self._input_rl_db, self._input_vswr = self._process_rl_vswr(
            input_rl_db, input_vswr, side="input"
        )

        # Output side
        self._output_rl_db, self._output_vswr = self._process_rl_vswr(
            output_rl_db, output_vswr, side="output"
        )

    def _process_rl_vswr(
        self,
        rl_db: Optional[Quantity],
        vswr: Optional[Quantity],
        side: str = "input",
    ) -> (Optional[Quantity], Optional[Quantity]):
        """
        Process return loss and VSWR parameters for one side.

        Returns a tuple of (rl_db, vswr).
        """
        out_rl: Optional[Quantity] = None
        out_vswr: Optional[Quantity] = None
        # Both provided: check consistency
        if rl_db is not None and vswr is not None:
            rl_q = rl_db if isinstance(rl_db, Quantity) else rl_db * u.dB
            vswr_q = vswr if isinstance(vswr, Quantity) else vswr * u.dimensionless
           
            if rl_q < 0 * u.dB:
                raise ValueError(
                    f"Stage '{self.label}': {side} return loss cannot be negative ({rl_q})."
                )
            if vswr_q < 1.0 * u.dimensionless:
                raise ValueError(
                    f"Stage '{self.label}': {side} VSWR must be >= 1 ({vswr_q})."
                )
               
            calc_vswr = return_loss_to_vswr(rl_q)
            if not math.isclose(vswr_q.value, calc_vswr.value, rel_tol=1e-4):
                raise ValueError(
                    f"Stage '{self.label}': {side} rl_db ({rl_q}) and vswr ({vswr_q}) "
                    f"are inconsistent (calc {calc_vswr})."
                )
            out_rl = rl_q
            out_vswr = vswr_q

        elif rl_db is not None:
            rl_q = rl_db if isinstance(rl_db, Quantity) else rl_db * u.dB
            if rl_q < 0 * u.dB:
                raise ValueError(
                    f"Stage '{self.label}': {side} return loss cannot be negative ({rl_q})."
                )
            out_rl = rl_q
            out_vswr = return_loss_to_vswr(out_rl)

        elif vswr is not None:
            vswr_q = vswr if isinstance(vswr, Quantity) else vswr * u.dimensionless
            if vswr_q < 1.0 * u.dimensionless:
                raise ValueError(
                    f"Stage '{self.label}': {side} VSWR must be >= 1 ({vswr_q})."
                )
            out_vswr = vswr_q
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
        # Convert from dB to linear using to_linear function
        lin_gain = to_linear(self._gain)
        # Round to integer if very close to avoid floating-point artifacts
        if abs(lin_gain.value - round(lin_gain.value)) < 1e-9:
            lin_gain = round(lin_gain.value) * u.dimensionless
        return lin_gain

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
        if nf is not None:
            return to_linear(nf)
        return None

    @property
    def p1db_dbm(self) -> Optional[Quantity]:
        """Output P1dB (Quantity with dBm unit)."""
        return self._p1db_dbm

    @property
    def p1db_mw(self) -> Optional[Quantity]:
        """Output P1dB (Quantity with mW unit)."""
        if self._p1db_dbm is not None:
            # Convert from dBm to mW
            return to_linear(self._p1db_dbm) * u.mW
        return None

    @property
    def input_rl_db(self) -> Optional[Quantity]:
        """Input return loss (Quantity with dB unit)."""
        return self._input_rl_db

    @property
    def output_rl_db(self) -> Optional[Quantity]:
        """Output return loss (Quantity with dB unit)."""
        return self._output_rl_db

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
        if self._gain.value >= 0:
            data["gain"] = float(self._gain.value)
        else:
            # loss is stored as negative gain
            data["loss"] = float((-self._gain).value)
        # Noise figure
        if self.noise_figure is not None:
            data["noise_figure"] = float(self.noise_figure.value)
        # P1dB
        if self._p1db_dbm is not None:
            data["p1db_dbm"] = float(self._p1db_dbm.value)
        # Return loss
        if self._input_rl_db is not None:
            data["input_rl_db"] = float(self._input_rl_db.value)
        if self._output_rl_db is not None:
            data["output_rl_db"] = float(self._output_rl_db.value)
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
        """Total cascaded gain in decibels, accounting for mismatch losses."""
        # No stages: zero gain
        if not self.stages:
            return 0.0 * u.dB
        # Sum stage gains (in dB)
        total_gain = sum(stage.gain for stage in self.stages)
        # Subtract mismatch losses between stages: for each stage after the first,
        # apply mismatch_loss using the stage input return loss (if specified).
        mismatch_loss_total = 0.0 * u.dB
        for stage in self.stages[1:]:
            rl = stage.input_rl_db
            if rl is not None:
                mismatch_loss_total += mismatch_loss(rl)
        return total_gain - mismatch_loss_total

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
        total_recip = 0.0
        gain_after = 1.0 * u.dimensionless
        for stage in reversed(self.stages):
            p1_q = stage.p1db_mw
            if p1_q is None:
                raise ValueError(f"Stage '{stage.label}' missing P1dB.")
            # Work in milliwatts
            total_recip += 1.0 / (p1_q.value * gain_after.value)
            # Accumulate linear gain
            gain_after *= stage.gain_lin
        if total_recip <= 0:
            return None
        out_p1_mw = 1.0 / total_recip
        # Convert to dBm quantity - manually calculate 10*log10(mW)
        out_p1_dbm = (10 * np.log10(out_p1_mw)) * u.dBm

        # Refer back to input (subtract total gain in dB)
        total_gain = self.cascaded_gain()
        input_p1_dbm = out_p1_dbm - total_gain
        return input_p1_dbm

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
