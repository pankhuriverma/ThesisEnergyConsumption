from pypapi import papi_high, events as papi_events
import numpy as np
import pandas as pd
import pyRAPL
from pypapi import papi_high
import time
pyRAPL.setup()

def matrix_multiplication(size):
    a = np.random.rand(size, size)
    b = np.random.rand(size, size)
    return np.dot(a, b)




# Starts counters
papi_high.ipc()  # -> Flops(0, 0, 0, 0)

matrix_multiplication(100)
result = papi_high.ipc()  # -> Flops(rtime, ptime, flpops, mflops)
print(result.ipc)
papi_high.stop_counters()   # -> []



