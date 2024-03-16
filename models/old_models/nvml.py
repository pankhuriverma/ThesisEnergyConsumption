"""import pynvml

pynvml.nvmlInit()
device_count = pynvml.nvmlDeviceGetCount()
for i in range(device_count):
    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
    device_name = pynvml.nvmlDeviceGetName(handle)
    print(f"Device {i}: {device_name.encode('utf-8')}")

pynvml.nvmlShutdown()
"""

import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
