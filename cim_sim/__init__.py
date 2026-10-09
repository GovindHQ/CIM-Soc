"""
cim_sim — a software model of a 32x32 weight-stationary CIM array, and a
drop-in replacement for the matmuls inside Vision Transformer attention.

Quick start
-----------
    from cim_sim import CIMArray, CIMConfig, TraceLog, cim_matmul

    array = CIMArray(CIMConfig(rows=32, cols=32, act_bits=4, weight_bits=4))
    log = TraceLog()
    Y = cim_matmul(X, W, array, tag="my_gemm", log=log)   # X[N,K] @ W[K,M]
    print(report(log))
"""

from .array import CIMArray, CIMConfig, CIMStats
from .matmul import MatmulTrace, TraceLog, cim_matmul, cim_matmul_batched
from .quant import cosine_sim, dequantize, fake_quant, quantize, rel_error
from .report import report, tile_map

__all__ = [
    "CIMArray", "CIMConfig", "CIMStats",
    "cim_matmul", "cim_matmul_batched", "TraceLog", "MatmulTrace",
    "quantize", "dequantize", "fake_quant", "rel_error", "cosine_sim",
    "report", "tile_map",
]