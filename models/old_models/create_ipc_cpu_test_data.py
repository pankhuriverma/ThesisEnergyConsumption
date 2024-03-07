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
import pyRAPL
import numpy as np
import matplotlib.pyplot as plt
import csv
import tensorflow as tf
import time
from sklearn.datasets import load_iris
from sklearn import tree

pyRAPL.setup()
pyRAPL.setup()


"""def model_initialise():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10),
        ]
    )
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    return x_train, y_train, model, loss_fn"""
"""def train_model(epochs_num, x_train, y_train, model, loss_fn):

    model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=epochs_num)

    return model"""


"""def measure_model_perf_energy(epochs_num, x_train, y_train, model, loss_fn):
        sum_ipc = 0
        sum_cpu_energy = 0
        sum_dram_energy = 0
        sum_total_energy = 0

        for index in range(1):
            meter = pyRAPL.Measurement('DL Model')
            meter.begin()
            papi_high.ipc()
            train_model(epochs_num, x_train, y_train, model, loss_fn)
            result = papi_high.ipc()
            papi_high.stop_counters()
            meter.end()
            output = meter.result
            cpu_energy = output.pkg[0]  # Assuming single-socket CPU; adjust as necessary
            dram_energy = output.dram[0]
            total_energy = cpu_energy + dram_energy

            sum_ipc += result.ipc
            sum_cpu_energy += cpu_energy
            sum_dram_energy += dram_energy
            sum_total_energy += total_energy




        # Calculate the averages of the energy measurements
        avg_ipc = sum_ipc
        avg_cpu_energy = sum_cpu_energy
        avg_dram_energy = sum_dram_energy
        avg_total_energy = sum_total_energy

        print(avg_ipc)
        print(avg_cpu_energy)
        print(avg_dram_energy)
        print(avg_total_energy)
        return avg_ipc, avg_cpu_energy, avg_dram_energy, avg_total_energy"""

if __name__ == "__main__":
    event_data = []
    total_energy = []
    cpu_energy = []
    dram_energy = []
    total_time = []
    counter = []
    all_events = {}


    energy_names = ["cpu_energy", "dram_energy", "total_energy"]

    num_of_epochs = list(range(1,50))

    x_train, y_train, model, loss_fn = model_initialise()
    cnt = 0
    for epochs_num in num_of_epochs:
        print(epochs_num)
        ipc, cpu_ene, dram_ene, total_ene = measure_model_perf_energy(epochs_num, x_train, y_train, model, loss_fn)

        event_data.append(ipc)
        cpu_energy.append(cpu_ene)
        dram_energy.append(dram_ene)
        total_energy.append(total_ene)

        cnt += 1
        counter.append(cnt)


    all_events["index"] = counter
    all_events["ipc"] = event_data
    all_events["cpu energy"] = cpu_energy
    all_events["dram energy"] = dram_energy
    all_events["total energy"] = total_energy



    df = pd.DataFrame(all_events)
    csv_file = '../../dataset/ipc_dataset/old_dataset/DL_model_test_dataset_ipc_10iterations_avg.csv'  # Specify your CSV file name
    df.to_csv(csv_file, index=False, mode = 'w')

