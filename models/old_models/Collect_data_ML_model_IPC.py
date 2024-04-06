# Import necessary libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from pypapi import papi_high, events as papi_events
import pyRAPL
import pandas as pd
import time
from pypapi import papi_high
pyRAPL.setup()

def measure_model_perf_energy(X_train, y_train):
        sum_ipc = 0
        sum_cpu_energy = 0
        sum_dram_energy = 0
        sum_total_energy = 0
        cnt = 0
        for index in range(10):
            meter = pyRAPL.Measurement('LR Model')
            meter.begin()
            papi_high.ipc()
            regr = LinearRegression()
            regr.fit(X_train, y_train)
            result = papi_high.ipc()
            papi_high.stop_counters()
            meter.end()
            output = meter.result
            cpu_energy = output.pkg[0] / 1000000 # Assuming single-socket CPU; adjust as necessary
            dram_energy = output.dram[0] /1000000
            total_energy = cpu_energy + dram_energy

            sum_ipc += result.ipc
            sum_cpu_energy += cpu_energy
            sum_dram_energy += dram_energy
            sum_total_energy += total_energy
            cnt += cnt
            print(cnt)



        # Calculate the averages of the energy measurements
        avg_ipc = sum_ipc / 10
        avg_cpu_energy = sum_cpu_energy / 10
        avg_dram_energy = sum_dram_energy / 10
        avg_total_energy = sum_total_energy / 10


        return avg_ipc, avg_cpu_energy, avg_dram_energy, avg_total_energy

if __name__ == "__main__":
    event_data = []
    total_cycles = []
    total_energy = []
    cpu_energy = []
    dram_energy = []
    total_time = []
    counter = []
    all_events = {}


    energy_names = ["cpu_energy", "dram_energy", "total_energy"]

    # Fetch the California housing dataset
    california_housing = datasets.fetch_california_housing()
    X, y = california_housing.data, california_housing.target
    cnt = 0
    print("length of x:",  len(X))
    for i in range(8000,9000):  # X.shape[1] gives the number of columns (features)
                # Select the current feature
                X_train = X[:i, :]
                y_train = y[:i]   # Selecting a single feature column; reshaping is not required here

                # Split the data into training/testing sets
                X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

                ipc, cpu_ene, dram_ene, total_ene = measure_model_perf_energy(X_train, y_train)

                event_data.append(ipc)
                cpu_energy.append(cpu_ene)
                dram_energy.append(dram_ene)
                total_energy.append(total_ene)

                cnt += 1
                counter.append(cnt)
                print(cnt)

    all_events["index"] = counter
    all_events["ipc"] = event_data
    all_events["cpu energy"] = cpu_energy
    all_events["dram energy"] = dram_energy
    all_events["total energy"] = total_energy




    df = pd.DataFrame(all_events)
    csv_file = '../../dataset/ipc_dataset/old_dataset/ML_model_linear_test_ipc_10iterations_avg.csv'  # Specify your CSV file name
    df.to_csv(csv_file, index=False, mode = 'w')

