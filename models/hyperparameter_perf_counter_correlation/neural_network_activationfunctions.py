import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import os
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


def plot_3_graphs(activation_functions, ipc_data, total_cycles, cpu_energy, dram_energy, total_energy, accuracies):
    # Create 1x3 subplots
    fig, axs = plt.subplots(2, 3, figsize=(10, 5))

    # Plot IPC
    axs[0,0].plot(activation_functions, ipc_data, marker='o', color='red')
    axs[0,0].set_title('Instructions vs Activation Functions')
    axs[0,0].set_xlabel('Activation Functions')
    axs[0,0].set_ylabel('Instructions')

    # Plot Cycles
    axs[0,1].plot(activation_functions, total_cycles, marker='o', color='blue')
    axs[0,1].set_title('Cycles vs Activation Functions')
    axs[0,1].set_xlabel('Activation Functions')
    axs[0,1].set_ylabel('Cycles')

    # Plot Accuracy
    axs[0,2].plot(activation_functions, accuracies, marker='o', color='green')
    axs[0,2].set_title('Accuracy vs Activation Functions')
    axs[0,2].set_xlabel('Activation Functions')
    axs[0,2].set_ylabel('Accuracy')

    # Plot CPU Energy
    axs[1,0].plot(activation_functions, cpu_energy, marker='o', color='red')
    axs[1,0].set_title('CPU Energy vs Activation Functions')
    axs[1,0].set_xlabel('Activation Functions')
    axs[1,0].set_ylabel('CPU Energy')

    # Plot DRAM Energy
    axs[1,1].plot(activation_functions, dram_energy, marker='o', color='blue')
    axs[1,1].set_title('DRAM Energy vs Activation Functions')
    axs[1,1].set_xlabel('Activation Functions')
    axs[1,1].set_ylabel('DRAM Energy')

    axs[1, 2].axis('off')
    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Display the plots
    plt.show()


# Load the breast cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Define a function to create the model
def create_model(activation):
    model = Sequential([
        Dense(16, activation=activation, input_shape=(X_train_scaled.shape[1],)),
        Dense(16, activation=activation),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# List of activation functions to try
activation_functions = [ 'tanh',  'softmax',  'sigmoid', 'relu']
#activation_functions = ['leaky_relu', 'relu', 'tanh', 'sigmoid', 'softmax', 'softplus']

# Initialize a list to store the accuracy of each model
accuracies = []
total_instructions = []
total_cycles = []
cpu_energy = []
dram_energy = []
total_energy = []
data = {}

# Loop over learning rates
for activation in activation_functions:
    model = create_model(activation=activation)
    meter = pyRAPL.Measurement('LR Model')
    meter.begin()
    papi_high.start_counters([papi_events.PAPI_TOT_INS, papi_events.PAPI_TOT_CYC])
    # Train the model
    model.fit(X_train_scaled, y_train, epochs=100, batch_size=30, verbose=0)

    counters = papi_high.stop_counters()
    meter.end()
    ins = counters[0]
    cycle = counters[1]
    total_instructions.append(ins)
    total_cycles.append(cycle)

    output = meter.result
    cpu_ener = output.pkg[0] / 1000000  # Assuming single-socket CPU; adjust as necessary
    dram_ener = output.dram[0] / 1000000
    total_ener = cpu_ener + dram_ener

    cpu_energy.append(cpu_ener)
    dram_energy.append(dram_ener)
    total_energy.append(total_ener)
    # Evaluate the model on the test set
    loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    accuracies.append(accuracy)
    print(f"Activation functions: {activation}, Test Accuracy: {accuracy:.4f}")


data["activation function"] = activation_functions
data["ins"] = total_instructions
data["cycles"] = total_cycles
data["cpu energy"] = cpu_energy
data["dram energy"] = dram_energy
data["accuracies"] = accuracies

print(activation_functions)
print(total_instructions)
print(total_cycles)

df = pd.DataFrame(data)
#csv_file = '../dataset/hyperparameter_dataset/NN_model_batchsize.csv' # Specify your CSV file name
csv_file = "/home/pankhuri/PycharmProjects/ThesisProject/dataset/hyperparameter_dataset/NN_model_activation_functions.csv"
df.to_csv(csv_file, index=False, mode = 'w')

plot_3_graphs(activation_functions, total_instructions, total_cycles, cpu_energy, dram_energy, total_energy, accuracies)
