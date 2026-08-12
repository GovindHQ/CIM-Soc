from cim_sim.quant import quantize, dequantize
import numpy as np

x = np.array([0.4, -1.1, 2.8, -0.7])

q, s = quantize(x, bits=4)

print("Original:", x)
print("Quantized:", q)
print("Scale:", s)

print("Recovered:", dequantize(q, s))