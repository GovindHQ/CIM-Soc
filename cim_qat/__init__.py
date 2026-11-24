"""cim_qat — hardware-aware QAT for TinyViT-5M against the cim_sim analog model.

The simulator (cim_sim) is the forward reference; this package holds the
differentiable surrogate, the calibration procedure, data handling and the
training/evaluation drivers. Nothing here modifies cim_sim.
"""
