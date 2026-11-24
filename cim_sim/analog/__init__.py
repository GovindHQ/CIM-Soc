"""
cim_sim.analog — physical analog datapath for one CIM macro.

    X_q -> input DAC -> V_IA -> C2C macro -> V_OA -> pre-ADC gain -> V_ADC_IN
        -> SAR ADC -> signed code d -> digital accumulation / rescale (matmul.py)

Opt-in only: CIMArray builds an AnalogChain when
`CIMConfig(analog=AnalogConfig(enabled=True, ...))`. With `analog=None` (the
default) or `enabled=False`, the legacy digital-MAC path and its functional
`adc_enabled` ADC model in array.py are used unchanged.

Quick start
-----------
    from cim_sim import CIMArray, CIMConfig
    from cim_sim.analog import AnalogConfig

    cfg = CIMConfig(weight_bits=8, act_bits=8,
                    analog=AnalogConfig(enabled=True,
                                        gain_per_op={"qkv": 4.0, "qk": 2.0}))
    array = CIMArray(cfg)
    print(array.chain.scale_report())
"""

from .adc import (ADCNonidealities, GainNonidealities, PreADCGain, SARADC,
                  SARDAC, SARDACNonidealities)
from .chain import (DAC_ALIGNMENTS, QAT_INITIAL_ADC_BITS, AnalogChain,
                    AnalogConfig, AnalogStats)
from .dac import CapacitiveDAC, DACNonidealities
from .digital import DigitalAccumulator, Requantizer
from .macro import C2CMacro, MacroNonidealities, SignedWeightAdapter

__all__ = [
    # chain
    "AnalogChain", "AnalogConfig", "AnalogStats", "DAC_ALIGNMENTS",
    "QAT_INITIAL_ADC_BITS",
    # input DAC (hardware-specified unsigned 8-bit capacitive DAC only)
    "CapacitiveDAC", "DACNonidealities",
    # C2C macro
    "C2CMacro", "SignedWeightAdapter", "MacroNonidealities",
    # gain + SAR ADC
    "PreADCGain", "SARADC", "SARDAC",
    "GainNonidealities", "ADCNonidealities", "SARDACNonidealities",
    # digital back end
    "DigitalAccumulator", "Requantizer",
]
