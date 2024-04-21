import numpy as np
import os
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
from pypapi import papi_high, events as papi_events
import numpy as np
import tensorflow as tf
import random as python_random
import pyRAPL
pyRAPL.setup()
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
def reset_random_seeds():
   np.random.seed(42)
   python_random.seed(42)
   tf.random.set_seed(42)

reset_random_seeds()


def plot_3_graphs(dropout_rates, ipc_data, total_cycles, cpu_energy, dram_energy, accuracies):
    # Create 1x3 subplots
    fig, axs = plt.subplots(2, 3, figsize=(15, 5))

    # Plot IPC
    axs[0,0].plot(dropout_rates, ipc_data, marker='o', color='red')
    axs[0,0].set_title('Instructions vs Dropout Rates')
    axs[0,0].set_xlabel('Dropout Rate')
    axs[0,0].set_ylabel('Instructions')

    # Plot Cycles
    axs[0,1].plot(dropout_rates, total_cycles, marker='o', color='blue')
    axs[0,1].set_title('Cycles vs Dropout Rates')
    axs[0,1].set_xlabel('Dropout Rate')
    axs[0,1].set_ylabel('Cycles')

    # Plot Accuracy
    axs[0,2].plot(dropout_rates, accuracies, marker='o', color='green')
    axs[0,2].set_title('Accuracy vs Dropout Rates')
    axs[0,2].set_xlabel('Dropout Rate')
    axs[0,2].set_ylabel('Accuracy')

    # Plot CPU Energy
    axs[1, 0].plot(dropout_rates, cpu_energy, marker='o', color='red')
    axs[1, 0].set_title('CPU Energy vs Dropout Rate')
    axs[1, 0].set_xlabel('Dropout Rate')
    axs[1, 0].set_ylabel('CPU Energy')

    # Plot DRAM Energy
    axs[1, 1].plot(dropout_rates, dram_energy, marker='o', color='blue')
    axs[1, 1].set_title('DRAM Energy vs Dropout Rate')
    axs[1, 1].set_xlabel('Dropout Rate')
    axs[1, 1].set_ylabel('DRAM Energy')

    axs[1, 2].axis('off')
    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Display the plots
    plt.show()

# Load and prepare the breast cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def create_model(dropout_rate):
    model = Sequential([
        Dense(16, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dropout(dropout_rate),
        Dense(16, activation='relu'),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

#dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # Define the dropout rates to experiment with
dropout_rates = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
total_instructions = []
total_cycles = []
cpu_energy = []
dram_energy = []
accuracies = []
data = {}

# Loop over the dropout rates
for rate in dropout_rates:
    model = create_model(dropout_rate=rate)
    meter = pyRAPL.Measurement('LR Model')
    meter.begin()
    papi_high.start_counters([papi_events.PAPI_TOT_INS, papi_events.PAPI_TOT_CYC])

    # Train the model
    model.fit(X_train_scaled, y_train, epochs=100, batch_size=10, verbose=0)
    counters = papi_high.stop_counters()
    meter.end()
    ins = counters[0]
    cycle = counters[1]
    total_instructions.append(ins)
    total_cycles.append(cycle)
    output = meter.result
    cpu_ener = output.pkg[0] / 1000000  # Assuming single-socket CPU; adjust as necessary
    dram_ener = output.dram[0] / 1000000

    cpu_energy.append(cpu_ener)
    dram_energy.append(dram_ener)
    # Optionally, evaluate the model's accuracy on the test set
    loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    accuracies.append(accuracy)
    print(f"Dropout rate: {rate}, Test Accuracy: {accuracy:.4f}")

data["dropout rate"] = dropout_rates
data["ins"] = total_instructions
data["cycles"] = total_cycles
data["cpu energy"] = cpu_energy
data["dram energy"] = dram_energy
data["accuracies"] = accuracies

print(dropout_rates)
print(total_instructions)
print(total_cycles)

df = pd.DataFrame(data)
#csv_file = '../dataset/hyperparameter_dataset/NN_model_batchsize.csv' # Specify your CSV file name
csv_file = "/home/pankhuri/PycharmProjects/ThesisProject/dataset/hyperparameter_dataset/NN_model_dropout_rate.csv"
df.to_csv(csv_file, index=False, mode = 'w')

#plot_graph(dropout_rates, total_cycles, accuracies)
plot_3_graphs(dropout_rates, total_instructions, total_cycles, cpu_energy, dram_energy, accuracies)