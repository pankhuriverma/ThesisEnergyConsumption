"""import tensorflow as tf

# Check for GPU availability
print(tf.config.experimental.list_physical_devices())
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
if tf.config.experimental.list_physical_devices('GPU'):
    print("GPU is available.")
else:
    print("GPU is not available. Please check your installation.")



"""
"""
import pycuda.driver as cuda
import pycuda.autoinit  # This is needed for initializing CUDA driver


def main():
    # Initialize CUDA Device
    # cuda.init() # autoinit takes care of this

    # Get count of GPUs available
    gpu_count = cuda.Device.count()

    print(f"Number of GPUs available: {gpu_count}")

    # Loop through GPUs and print their details
    for i in range(gpu_count):
        gpu = cuda.Device(i)
        print(f"\nGPU {i} - {gpu.name()}")
        print(f"  Compute Capability: {gpu.compute_capability()}")
        print(f"  Total Memory: {gpu.total_memory() // (1024 ** 2)} MB")
        # You can query more details about the GPU as needed


if __name__ == "__main__":
    main()"""


import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np
import time
from pycuda.curandom import rand as curand
from pycuda.elementwise import ElementwiseKernel

# Define the size of the matrices
MATRIX_SIZE = 10240

# Generate random matrices (for example purposes)
a_cpu = np.random.randn(MATRIX_SIZE, MATRIX_SIZE).astype(np.float32)
b_cpu = np.random.randn(MATRIX_SIZE, MATRIX_SIZE).astype(np.float32)

# Transfer matrices to GPU
a_gpu = gpuarray.to_gpu(a_cpu)
b_gpu = gpuarray.to_gpu(b_cpu)

# Allocate memory for the result
c_gpu = gpuarray.empty((MATRIX_SIZE, MATRIX_SIZE), np.float32)

# Create an ElementwiseKernel for matrix multiplication
matrix_mul_kernel = ElementwiseKernel(
    "float *a, float *b, float *c",
    "c[i] = a[i] * b[i]",
    "matrix_mul_kernel")

start_time = time.time()

# Perform matrix multiplication
matrix_mul_kernel(a_gpu, b_gpu, c_gpu)

elapsed_time = time.time() - start_time

# Print the time taken
print(f"Time taken for matrix multiplication: {elapsed_time} seconds")

# Copy the result back to CPU
c_cpu = c_gpu.get()

# You can now print or use the resulting matrix c_cpu"""
"""
import pycuda.driver as cuda
import pycuda.autoinit  # Automatically initializes CUDA


def print_gpu_info():
    # Get the number of GPUs available
    num_gpus = cuda.Device.count()

    print(f"Number of GPUs: {num_gpus}")

    for i in range(num_gpus):
        gpu = cuda.Device(i)
        attributes = gpu.get_attributes()

        # Various device attributes
        name = gpu.name()  # Name of the GPU
        total_memory = gpu.total_memory() // (1024 ** 2)  # Convert bytes to MB
        compute_capability = f"{gpu.compute_capability()[0]}.{gpu.compute_capability()[1]}"

        print(f"\nGPU {i}: {name}")
        print(f"Total Memory: {total_memory} MB")
        print(f"Compute Capability: {compute_capability}")

        # Print more attributes as needed
        for key, value in attributes.items():
            print(f"{cuda.device_attribute_to_string(key)}: {value}")


print_gpu_info()"""




