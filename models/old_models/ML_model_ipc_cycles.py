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

        meter = pyRAPL.Measurement('LR Model')
        meter.begin()
        start_time = time.time()
        papi_high.start_counters([papi_events.PAPI_TOT_CYC])
        papi_high.ipc()

        regr = LinearRegression()
        regr.fit(X_train, y_train)
        result = papi_high.ipc()
        papi_high.stop_counters()
        counters = papi_high.stop_counters()



        end_time = time.time()
        meter.end()
        tot_time = end_time - start_time
        output = meter.result
        cpu_energy = output.pkg[0]  # Assuming single-socket CPU; adjust as necessary
        dram_energy = output.dram[0]
        total_energy = cpu_energy + dram_energy

        return result.ipc, counters[0], cpu_energy, dram_energy, total_energy, tot_time

if __name__ == "__main__":
    event_data = []
    total_cycles = []
    total_energy = []
    cpu_energy = []
    dram_energy = []
    total_time = []
    counter = []
    all_events = {}
    ipc_data = []

    # Fetch the California housing dataset
    california_housing = datasets.fetch_california_housing()
    X, y = california_housing.data, california_housing.target
    cnt = 0
    print("length of x:",  len(X))
    for i in range(10000, 20460):  # X.shape[1] gives the number of columns (features)
                # Select the current feature
                X_train = X[:i, :]
                y_train = y[:i]   # Selecting a single feature column; reshaping is not required here

                # Split the data into training/testing sets
                X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

                ipc, cyc, cpu_ene, dram_ene, total_ene, tot_time = measure_model_perf_energy(X_train, y_train)
                event_data.append(cyc)
                ipc_data.append(ipc)
                cpu_energy.append(cpu_ene)
                dram_energy.append(dram_ene)
                total_energy.append(total_ene)
                total_time.append(tot_time)
                cnt += 1
                counter.append(cnt)
                print(cnt)

    all_events["index"] = counter
    all_events["ipc"] = ipc_data
    all_events["cycles"] = event_data
    all_events["cpu energy"] = cpu_energy
    all_events["dram energy"] = dram_energy
    all_events["total energy"] = total_energy
    all_events["total time"] = total_time



    df = pd.DataFrame(all_events)
    csv_file = '../dataset/ipc_cycles_dataset/ML_model_collected_dataset_ipc_cycles.csv'  # Specify your CSV file name
    df.to_csv(csv_file, index=False, mode = 'a')

