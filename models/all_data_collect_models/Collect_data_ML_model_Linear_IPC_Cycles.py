# Import necessary libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from pypapi import papi_high, events as papi_events
import pyRAPL
import pandas as pd
import time
from pypapi import papi_high, events as papi_events
pyRAPL.setup()

def measure_model_perf_energy(X_train, y_train):
    sum_ipc = 0
    sum_ins = 0
    sum_cycles = 0
    sum_cpu_energy = 0
    sum_dram_energy = 0
    sum_total_energy = 0
    cnt = 0
    for index in range(10):
        meter = pyRAPL.Measurement('LR Model')
        meter.begin()

        papi_high.start_counters([papi_events.PAPI_TOT_INS, papi_events.PAPI_TOT_CYC])
        regr = LinearRegression()
        regr.fit(X_train, y_train)
        counters = papi_high.stop_counters()
        meter.end()
        ins = counters[0]
        cycle = counters[1]
        ipc = ins / cycle if cycle > 0 else 0

        output = meter.result
        cpu_ener = output.pkg[0] / 1000000 # Assuming single-socket CPU; adjust as necessary
        dram_ener = output.dram[0] / 1000000
        total_ener = cpu_ener + dram_ener

        sum_ins += ins
        sum_ipc += ipc
        sum_cycles += cycle
        sum_cpu_energy += cpu_ener
        sum_dram_energy += dram_ener
        sum_total_energy += total_ener
        cnt += cnt
        print(cnt)

    # Calculate the averages of the energy measurements
    avg_ins = sum_ins / 10
    avg_ipc = sum_ipc / 10
    avg_cycle = sum_cycles / 10
    avg_cpu_energy = sum_cpu_energy / 10
    avg_dram_energy = sum_dram_energy / 10
    avg_total_energy = sum_total_energy / 10

    return avg_ins, avg_cycle, avg_ipc, avg_cpu_energy, avg_dram_energy, avg_total_energy

if __name__ == "__main__":
    cyc_data = []
    ins_data = []
    ipc_data = []
    total_cycles = []
    total_energy = []
    cpu_energy = []
    dram_energy = []
    total_time = []
    counter = []
    all_events = {}

    # Fetch the California housing dataset
    #california_housing = datasets.fetch_california_housing()
    diabetes = datasets.load_diabetes()
    X, y = diabetes.data, diabetes.target
    cnt = 0
    print("length of x:",  len(X))
    for i in range(100, 443):  # X.shape[1] gives the number of columns (features)
                # Select the current feature
                X_train = X[:i, :]
                y_train = y[:i]   # Selecting a single feature column; reshaping is not required here

                # Split the data into training/testing sets
                X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

                ins, cyc, ipc, cpu_ene, dram_ene, total_ene = measure_model_perf_energy(X_train, y_train)
                ins_data.append(ins)
                ipc_data.append(ipc)
                cyc_data.append(cyc)

                cpu_energy.append(cpu_ene)
                dram_energy.append(dram_ene)
                total_energy.append(total_ene)

                cnt += 1
                counter.append(cnt)
                print(cnt)

    all_events["index"] = counter
    all_events["ins"] = ins_data
    all_events["cycles"] = cyc_data
    all_events["ipc"] = ipc_data
    all_events["cpu energy"] = cpu_energy
    all_events["dram energy"] = dram_energy
    all_events["total energy"] = total_energy




    df = pd.DataFrame(all_events)
    csv_file = '../../dataset/ipc_cycles_dataset/ML_model_linear_dataset.csv'  # Specify your CSV file name
    df.to_csv(csv_file, index=False, mode = 'w')

