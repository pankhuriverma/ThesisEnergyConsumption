# Import necessary libraries
import pyRAPL
import pandas as pd
import time
from pypapi import papi_high, events as papi_events

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import os
import numpy as np
import tensorflow as tf
import random as python_random
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
def reset_random_seeds():
   np.random.seed(42)
   python_random.seed(42)
   tf.random.set_seed(42)

reset_random_seeds()

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
        history = model.fit(X_train, y_train, epochs=10, batch_size=10, verbose=1)
        counters = papi_high.stop_counters()
        ins = counters[0]
        cycle = counters[1]
        ipc = ins / cycle if cycle > 0 else 0

        meter.end()

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


    # Load the breast cancer dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target
    print(X.shape)
    print(y.shape)

    cnt = 0
    print("length of x:",  len(X))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define the neural network model
    model = Sequential([
        Dense(16, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    for i in range(100, 455):  # X.shape[1] gives the number of columns (features)

        X_train_data = X_train_scaled[0:i, :]
        y_train_data = y_train[0:i]
        print(X_train_data.shape)
        print(y_train_data.shape)

        ins, cyc, ipc, cpu_ene, dram_ene, total_ene = measure_model_perf_energy(X_train_data, y_train_data)
        ins_data.append(ins)
        ipc_data.append(ipc)
        cyc_data.append(cyc)

        cpu_energy.append(cpu_ene)
        dram_energy.append(dram_ene)
        total_energy.append(total_ene)

        cnt += 1
        counter.append(cnt)
        print(cnt)
        loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=1)
        print(f'Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')

    all_events["index"] = counter
    all_events["ins"] = ins_data
    all_events["cycles"] = cyc_data
    all_events["ipc"] = ipc_data
    all_events["cpu energy"] = cpu_energy
    all_events["dram energy"] = dram_energy
    all_events["total energy"] = total_energy




    df = pd.DataFrame(all_events)
    csv_file = '../../dataset/ipc_cycles_dataset/old_dataset/NN_model_ipc_cycles_dataset_10_iterations_avg.csv'  # Specify your CSV file name
    df.to_csv(csv_file, index=False, mode = 'w')

