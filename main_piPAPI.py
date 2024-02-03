from pypapi import papi_high
from pypapi import events as papi_events
import numpy as np
import pandas as pd
import csv

def matrix_multiplication(size):
    a = np.random.rand(size, size)
    b = np.random.rand(size, size)
    result = np.dot(a, b)
    return result


def measure_performance(size):
    papi_output = []
    papi_high.start_counters([papi_events.PAPI_TOT_INS])
    matrix_multiplication(size)
    counters = papi_high.stop_counters()
    return counters[0]

if __name__ == "__main__":

    total_cycles = []
    total_instructions = []
    flops = []
    matrix_size = list(range(500, 502))
    print(len(matrix_size))


    for i in matrix_size:
        print(i)
        tot_ins = measure_performance(i)
        total_instructions.append(tot_ins)






    """with open('piPAPI_output.csv', 'a', newline='') as file:
        writer = csv.writer(file)

        writer.writerow(['Matrix Size', 'FLOPS'])
        for size, fl in zip(matrix_size, flops):
            writer.writerow([size, fl])
"""
    """plt.figure(figsize=(10, 6))
    plt.plot(matrix_size, flops, label='flops')

    # Adding title and labels
    plt.title('Matrix Size vs Flops')
    plt.xlabel('Matrix Size')
    plt.ylabel('flops')

    # Adding a grid for better readability
    plt.grid(True)

    # Showing the plot
    plt.legend()
    plt.show()"""

