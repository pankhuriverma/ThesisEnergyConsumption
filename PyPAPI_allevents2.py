from pypapi import papi_high, events as papi_events
import numpy as np
import pandas as pd
import pyRAPL
import time

pyRAPL.setup()


def matrix_multiplication(size):
    a = np.random.rand(size, size)
    b = np.random.rand(size, size)
    return np.dot(a, b)


def measure_performance_and_energy(size, event_name):
    # Setup pyRAPL measurement but don't start it yet
    pyRAPL.setup()
    meter = pyRAPL.Measurement('Matrix Multiplication')

    # Begin pyRAPL measurement
    meter.begin()

    # Start PAPI counters immediately before the operation
    papi_high.start_counters([event_name])

    # Perform the matrix multiplication
    start_time = time.time()
    matrix_multiplication(size)
    end_time = time.time()

    # Stop PAPI counters immediately after the operation
    counters = papi_high.stop_counters()

    # End pyRAPL measurement
    meter.end()

    # Calculate total time
    tot_time = end_time - start_time

    # Extract energy measurements from pyRAPL
    output = meter.result
    cpu_energy = output.pkg[0]  # Assuming single-socket CPU; adjust as necessary
    dram_energy = output.dram[0]
    total_energy = cpu_energy + dram_energy

    return counters[0], cpu_energy, dram_energy, total_energy, tot_time


if __name__ == "__main__":
    matrix_sizes = list(range(2000, 2010))
    all_data = []

    events = [papi_events.PAPI_TOT_INS, papi_events.PAPI_TOT_CYC, papi_events.PAPI_L1_TCM]
    event_names = ["Total Instructions", "Total Cycles", "L1 Cache Misses", "CPU Energy", "DRAM Energy", "Total Energy",
                   "Total Time"]

    for size in matrix_sizes:
        row_data = {"Matrix Size": size}
        for event, event_name in zip(events, event_names[:3]):
            performance_result, cpu_ene, dram_ene, total_ene, tot_time = measure_performance_and_energy(size, event)
            row_data.update({
                event_name: performance_result,
                "CPU Energy": cpu_ene,
                "DRAM Energy": dram_ene,
                "Total Energy": total_ene,
                "Total Time": tot_time
            })
        all_data.append(row_data)

    df = pd.DataFrame(all_data)
    csv_file = 'output_combined.csv'
    df.to_csv(csv_file, index=False)
