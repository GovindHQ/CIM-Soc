"""
cim_qat/hw_config.py — the ONE place where the QAT side reads hardware
parameters, and it reads them from the simulator.

Fixed hardware (not configurable here): capacitive input DAC 8 b @ 0.6 V
(cim_sim's only DAC), MSB alignment of narrower activations (simulator
default, provisional), 32x32 macro, m = 32, SAR ADC with V_REF 0.6 V.
Fixed for the initial QAT phase: ADC = 8 bits (cim_sim QAT_INITIAL_ADC_BITS).
Calibration parameters: the three per-operation gains {qkv, qk, av}.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .surrogate import TIE_GUARD_REL, OpSpec

OPS = ("qkv", "qk", "av")
OP_SIGNED = {"qkv": True, "qk": True, "av": False}


@dataclass(frozen=True)
class HWConfig:
    weight_bits: int
    act_bits: int
    gains: tuple            # (G_qkv, G_qk, G_av) — calibration parameters
    adc_bits: int = 8
    adc_vref: float = 0.6
    dac_bits: int = 8
    dac_vref: float = 0.6
    dac_input_alignment: str = "msb"
    rows: int = 32
    cols: int = 32

    @property
    def gain_dict(self) -> dict:
        return dict(zip(OPS, self.gains))

    @property
    def label(self) -> str:
        return f"W{self.weight_bits}A{self.act_bits}_ADC{self.adc_bits}"

    def __post_init__(self):
        object.__setattr__(self, "gains", tuple(float(g) for g in self.gains))
        if len(self.gains) != 3:
            raise ValueError("gains must be (G_qkv, G_qk, G_av)")
        if self.adc_bits != 8:
            raise ValueError("this QAT phase fixes the ADC at 8 bits (QAT_INITIAL_ADC_BITS)")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["gains"] = self.gain_dict
        d["dac_model"] = "capacitive (cim_sim.analog.CapacitiveDAC, the only DAC)"
        return d

    def job_dict(self) -> dict:
        """Plain constructor arguments (round-trips through JSON)."""
        d = asdict(self)
        d["gains"] = list(self.gains)
        return d

    # -- simulator objects ------------------------------------------------- #

    def analog_config(self):
        from cim_sim.analog import AnalogConfig
        return AnalogConfig(enabled=True, dac_bits=self.dac_bits,
                            dac_vref=self.dac_vref,
                            dac_input_alignment=self.dac_input_alignment,
                            gain_default=1.0, gain_per_op=self.gain_dict,
                            adc_bits=self.adc_bits, adc_vref=self.adc_vref)

    def cim_config(self, token_block=None):
        from cim_sim import CIMConfig
        return CIMConfig(rows=self.rows, cols=self.cols,
                         weight_bits=self.weight_bits, act_bits=self.act_bits,
                         token_block=token_block, analog=self.analog_config())


def op_specs(hw: HWConfig) -> dict:
    """{op: OpSpec} with LSB_P taken from the simulator's AnalogChain."""
    from cim_sim.analog import AnalogChain, SARADC
    assert SARADC.TIE_GUARD_REL == TIE_GUARD_REL, \
        "surrogate tie guard drifted from the simulator's SARADC.TIE_GUARD_REL"
    out = {}
    for op in OPS:
        ch = AnalogChain(hw.analog_config(), hw.rows, hw.weight_bits, hw.act_bits)
        ch.set_operation(op, act_signed=OP_SIGNED[op])
        out[op] = OpSpec(op=op, weight_bits=hw.weight_bits, act_bits=hw.act_bits,
                         act_signed=OP_SIGNED[op], lsb_p=float(ch.lsb_p),
                         adc_bits=int(ch.adc.bits), rows=hw.rows)
    return out


# --------------------------------------------------------------------------- #
# frozen gains (written only by `qat_calibrate.py freeze`)
# --------------------------------------------------------------------------- #

FROZEN_SCHEMA = "cim_qat.frozen_gains.v1"


def load_frozen(path, weight_bits: int, act_bits: int) -> tuple:
    """-> (HWConfig, record). Refuses a precision mismatch or a foreign file."""
    import json
    from pathlib import Path
    rec = json.loads(Path(path).read_text())
    if rec.get("schema") != FROZEN_SCHEMA:
        raise RuntimeError(f"{path}: not a frozen-gains file")
    if (rec["weight_bits"], rec["act_bits"]) != (weight_bits, act_bits):
        raise RuntimeError(f"{path}: gains were calibrated for W{rec['weight_bits']}A{rec['act_bits']}, "
                           f"run is W{weight_bits}A{act_bits}")
    g = rec["gains"]
    hw = HWConfig(weight_bits=weight_bits, act_bits=act_bits,
                  gains=(g["qkv"], g["qk"], g["av"]), adc_bits=rec["adc_bits"])
    return hw, rec
