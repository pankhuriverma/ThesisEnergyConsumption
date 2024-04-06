import numpy as np
import os
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from pypapi import papi_high, events as papi_events
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import random as python_random

# Ensure TensorFlow does not use GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def reset_random_seeds():
   np.random.seed(42)
   python_random.seed(42)
   tf.random.set_seed(42)

reset_random_seeds()

def plot_graph(layers, ipc, cycles):


    # Create a figure and a set of subplots
    fig, ax1 = plt.subplots()

    # Plot the first set of data and set axis labels
    color = 'tab:red'
    ax1.set_xlabel("layers")
    ax1.set_ylabel("ipc", color=color)
    ax1.scatter(layers, ipc, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # Instantiate a second y-axis sharing the same x-axis
    ax2 = ax1.twinx()

    # Plot the second set of data with a different color
    color = 'tab:blue'
    ax2.set_ylabel("cycles", color=color)
    ax2.scatter(layers, cycles, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # Title and grid
    plt.title('Graph of Peformance Counters vs  Energy (Huber Loss)')
    ax1.grid(True)

    # Show the plot
    plt.show()

def plot_3_graphs(layers, ipc_data, total_cycles, accuracies):
    # Create 1x3 subplots
    fig, axs = plt.subplots(1, 3, figsize=(10, 5))

    # Plot IPC
    axs[0].plot(layers, ipc_data, marker='o', color='red')
    axs[0].set_title('IPC vs Layers')
    axs[0].set_xlabel('Layers')
    axs[0].set_ylabel('IPC')

    # Plot Cycles
    axs[1].plot(layers, total_cycles, marker='o', color='blue')
    axs[1].set_title('Cycles vs Layers')
    axs[1].set_xlabel('Layers')
    axs[1].set_ylabel('Cycles')

    # Plot Accuracy
    axs[2].plot(layers, accuracies, marker='o', color='green')
    axs[2].set_title('Accuracy vs Layers')
    axs[2].set_xlabel('Layers')
    axs[2].set_ylabel('Accuracy')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Display the plots
    plt.show()

# Load the dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize a list to store the accuracy of each model
accuracies = []
ipc_data = []
total_cycles = []
layers = []
data = {}

# Loop to create models with 1 to 20 layers
for num_layers in range(20, 0, -1):
    model = Sequential()
    model.add(Dense(16, activation='relu', input_shape=(X_train_scaled.shape[1],)))
    layers.append(num_layers)
    # Add hidden layers
    for _ in range(num_layers - 1):
        model.add(Dense(16, activation='relu'))

    # Output layer
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    papi_high.start_counters([papi_events.PAPI_TOT_INS, papi_events.PAPI_TOT_CYC])



    # Train the model
    model.fit(X_train_scaled, y_train, epochs=100, batch_size=10, verbose=0)
    counters = papi_high.stop_counters()

    ins = counters[0]
    cycle = counters[1]
    ipc = ins / cycle if cycle > 0 else 0
    ipc_data.append(ipc)
    total_cycles.append(cycle)
    # Evaluate the model
    loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    accuracies.append(accuracy)

    print(f'Model with {num_layers} layers: Test Accuracy = {accuracy * 100:.2f}%')

data["layers"] = layers
data["ipc"] = ipc_data
data["cycles"] = total_cycles
data["accuracy"] = accuracies

print(layers)
print(ipc_data)
print(total_cycles)
print(accuracies)

plot_3_graphs(layers, ipc_data, total_cycles, accuracies)

