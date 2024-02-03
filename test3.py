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

def measure_performance(size, event_name):
    papi_high.start_counters([event_name])
    matrix_multiplication(size)
    counters = papi_high.stop_counters()
    return counters[0]

def measure_energy(size):

    pyrapl_output =[]
    pyRAPL.setup()
    meter = pyRAPL.Measurement('Matrix Multiplication')

    meter.begin()
    start_time = time.time()
    matrix_multiplication(size)
    end_time = time.time()
    meter.end()
    tot_time = end_time - start_time
    output = meter.result
    cpu_energy = output.pkg
    dram_energy = output.dram
    total_energy = cpu_energy[0] + dram_energy[0]

    return cpu_energy[0], dram_energy[0], total_energy, tot_time



if __name__ == "__main__":
    matrix_sizes = list(range(2000, 2010))
    event_data = []
    total_cycles = []
    total_energy = []
    cpu_energy = []
    dram_energy = []
    total_time = []
    all_events = {}
    events = [papi_events.PAPI_TOT_INS, papi_events.PAPI_TOT_CYC, papi_events.PAPI_L1_TCM]
    event_names = ["Total Instructions", "Total Cycles", "L1 Cache Misses"]
    energy_names = ["cpu_energy", "dram_energy", "total_energy", "total time"]
    counter = 0
    for event, event_name in zip(events, event_names):

        for size in matrix_sizes:
            performance_result = measure_performance(size, event)
            event_data.append(performance_result)
            print(counter)
            if counter >= 1:
                continue
            print("Energy measured")
            cpu_ene, dram_ene, total_ene, tot_time = measure_energy(size)
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
    csv_file = 'output.csv'  # Specify your CSV file name
    df.to_csv(csv_file, index=False)



