# Import necessary libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import pynvml
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

def measure_model_perf_energy(X_train, y_train, handle, model):

    mem = 0
    gpu = 0
    cnt = 0

    for index in range(10):
        regr = LinearRegression()
        regr.fit(X_train, y_train)

        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_utilization = utilization.gpu
        memory_utilization = utilization.memory
        #memory_utilization = pynvml.nvmlDeviceGetPowerUsage(handle)

        gpu += gpu_utilization
        mem += memory_utilization

        cnt += cnt
        print(cnt)

    # Calculate the averages of the energy measurements
    avg_mem = mem / 10
    avg_gpu = gpu / 10


    return avg_gpu, avg_mem

if __name__ == "__main__":

    gpu_data = []
    mem_data = []
    counter = []
    all_events = {}



    # Initialize NVML
    pynvml.nvmlInit()

    # Get GPU device count
    device_count = pynvml.nvmlDeviceGetCount()

    # Assuming a single GPU is being used
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    print(handle)

    # Fetch the California housing dataset
    california_housing = datasets.fetch_california_housing()
    X, y = california_housing.data, california_housing.target
    cnt = 0
    print("length of x:",  len(X))
    np.random.seed(42)
    model = Sequential([
        Dense(50, activation='relu', input_shape=(64,8)),
        # Dense(20, activation='relu'),
        Dense(2)  # Output layer
    ])
    for i in range(20450, 20460):  # X.shape[1] gives the number of columns (features)
                # Select the current feature
                X_train = X[:i, :]
                y_train = y[:i]   # Selecting a single feature column; reshaping is not required here

                # Split the data into training/testing sets
                X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

                tot_gpu, tot_mem = measure_model_perf_energy(X_train, y_train, handle, model)
                gpu_data.append(tot_gpu)
                mem_data.append(tot_mem)


                cnt += 1
                counter.append(cnt)
                print(cnt)

    #all_events["index"] = counter
    all_events["gpu util"] = gpu_data
    all_events["memory util"] = mem_data
    print(all_events)

    df = pd.DataFrame(all_events)
    csv_file = '../dataset/gpu/ML_model_gpu_dataset_10_iterations_avg.csv'  # Specify your CSV file name
    df.to_csv(csv_file, index=False, mode = 'w')


