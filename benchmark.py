import pyRAPL
from pypapi import papi_high
from pypapi import events as papi_events
import numpy as np
import statistics
import time


pyRAPL.setup()


def matrix_multiplication(size):
    a = np.random.rand(size, size)
    b = np.random.rand(size, size)
    result = np.dot(a, b)
    return result


def measure_performance(size):
    papi_output = []
    papi_high.start_counters([papi_events.PAPI_TOT_INS, papi_events.PAPI_TOT_CYC])
    matrix_multiplication(size)
    counters = papi_high.stop_counters()
    return counters[0], counters[1]
def measure_energy(size):

    pyrapl_output =[]
    pyRAPL.setup()
    meter = pyRAPL.Measurement('benchmark')

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
    print(output)
    return cpu_energy, dram_energy, total_energy, tot_time


if __name__ == "__main__":

    energy = []
    instructions = []
    cycles = []
    time_total = []

    size = 200
    for i in range(1,100):

        cpu_energy, dram_energy, total_energy, total_time = measure_energy(size)
        energy.append(total_energy)
        time_total.append(total_time)

        ins, cyc = measure_performance(size)
        instructions.append(ins)
        cycles.append(cyc)

    print('Energy in microjoules: ', statistics.mean(energy))
    print('Energy in joules: ', statistics.mean(energy)/1000000)
    energy_joules = statistics.mean(energy)/1000000
    print('Total time in sec: ', statistics.mean(time_total))
    power = energy_joules / statistics.mean(time_total)
    print('Power in Watt: ', power)
    print('Instructions: ', statistics.mean(instructions))
    print('Cycles: ', statistics.mean(cycles))
    print('Power per ins: ', power/statistics.mean(instructions))
    print('Power per cycle: ', power/statistics.mean(cycles))





