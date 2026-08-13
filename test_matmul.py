import numpy as np

from cim_sim.array import CIMArray, CIMConfig
from cim_sim.matmul import cim_matmul

cfg = CIMConfig(rows=2, cols=2)

array = CIMArray(cfg)

X = np.array([
    [1,2,3],
    [4,5,6]
], dtype=float)

W = np.array([
    [1,2,3],
    [4,5,6],
    [7,8,9]
], dtype=float)

y = cim_matmul(X, W, array)

print(y)

print(array.stats)