from pypapi import papi_high, events as papi_events
import numpy as np
import pandas as pd
import pyRAPL
import time
import scipy.stats as stats
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
    matrix_sizes = list(range(3000, 5000))
    event_data = []
    total_cycles = []
    total_energy = []
    cpu_energy = []
    dram_energy = []
    total_time = []
    all_events = {}
    events = [papi_events.PAPI_TOT_INS, papi_events.PAPI_TOT_CYC, papi_events.PAPI_L2_TCM]
    event_names = ["Total Instructions", "Total Cycles", "L1 Cache Misses"]
    energy_names = ["cpu_energy", "dram_energy", "total_energy", "total time"]
    counter = 0
    all_events["Matrix Size"] = matrix_sizes
    for event, event_name in zip(events, event_names):

        for size in matrix_sizes:
            perf_result, cpu_ene, dram_ene, total_ene, tot_time = measure_performance_and_energy(size, event)
            event_data.append(perf_result)
            print(counter)
            if counter >= 1:
                continue
            print("Energy measured")
            cpu_energy.append(cpu_ene)
            dram_energy.append(dram_ene)
            total_energy.append(total_ene)
            total_time.append(tot_time)

        counter = counter + 1



        all_events[event_name] = event_data
        event_data = []
    all_events["cpu energy"] = cpu_energy
    all_events["dram energy"] = dram_energy
    all_events["total energy"] = total_energy
    all_events["total time"] = total_time



    df = pd.DataFrame(all_events)
    csv_file = 'output_run_all_3.csv'  # Specify your CSV file name
    df.to_csv(csv_file, index=False)





