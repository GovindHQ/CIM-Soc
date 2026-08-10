import numpy as np
from cim_sim.array import CIMArray, CIMConfig

#testing just 2x2 cim array
cfg = CIMConfig(rows=2, cols=2)

#weight to be loaded
W = np.array([
    [2,1],
    [4,4]
], dtype=np.int32)

#input
x = np.array([[1,2], [2, 4], [5 , 11]], dtype=np.int32)

#creating object from the class
array = CIMArray(cfg)

#load weight function
array.load_weight_tile(W)

#running the mvm function which returns the output
y = array.mvm(x)

print(y)
print(array.stats)