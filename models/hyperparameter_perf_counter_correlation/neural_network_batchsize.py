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
def plot_graph(batch_size, ipc, cycles):


    # Create a figure and a set of subplots
    fig, ax1 = plt.subplots()

    # Plot the first set of data and set axis labels
    color = 'tab:red'
    ax1.set_xlabel("layers")
    ax1.set_ylabel("ipc", color=color)
    ax1.scatter(batch_size, ipc, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # Instantiate a second y-axis sharing the same x-axis
    ax2 = ax1.twinx()

    # Plot the second set of data with a different color
    color = 'tab:blue'
    ax2.set_ylabel("cycles", color=color)
    ax2.scatter(batch_size, cycles, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # Title and grid
    plt.title('Graph of Peformance Counters vs  Energy (Huber Loss)')
    ax1.grid(True)

    # Show the plot
    plt.show()

def plot_3_graphs(batch_size, ipc_data, total_cycles, accuracies):
    # Create 1x3 subplots
    fig, axs = plt.subplots(1, 3, figsize=(10, 5))

    # Plot IPC
    axs[0].plot(batch_size, ipc_data, marker='o', color='red')
    axs[0].set_title('IPC vs Batch Size')
    axs[0].set_xlabel('Batch Size')
    axs[0].set_ylabel('IPC')

    # Plot Cycles
    axs[1].plot(batch_size, total_cycles, marker='o', color='blue')
    axs[1].set_title('Cycles vs Batch Size')
    axs[1].set_xlabel('Batch Size')
    axs[1].set_ylabel('Cycles')

    # Plot Accuracy
    axs[2].plot(batch_size, accuracies, marker='o', color='green')
    axs[2].set_title('Accuracy vs Batch Size')
    axs[2].set_xlabel('Batch Size')
    axs[2].set_ylabel('Accuracy')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Display the plots
    plt.show()

# Function to create a model
def create_model():
    model = Sequential([
        Dense(16, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

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

# Initialize a list to store the accuracy of each model
accuracies = []
ipc_data = []
total_cycles = []
batch_size = []
data = {}


# Loop over batch sizes from 1 to 30
for batch in range(50, 200):
    # Create a new instance of the model for each batch size
    model = create_model()
    batch_size.append(batch)
    papi_high.start_counters([papi_events.PAPI_TOT_INS, papi_events.PAPI_TOT_CYC])

    # Train the model
    model.fit(X_train_scaled, y_train, batch_size=batch, epochs=50, verbose=0)
    counters = papi_high.stop_counters()

    ins = counters[0]
    cycle = counters[1]
    ipc = ins / cycle if cycle > 0 else 0
    ipc_data.append(ipc)
    total_cycles.append(cycle)
    # Evaluate the model on the test set
    loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)

    print(f"Batch Size: {batch}, Test Accuracy: {accuracy:.4f}")
    accuracies.append(accuracy)

data["batch_size"] = batch_size
data["ipc"] = ipc_data
data["cycles"] = total_cycles

print(batch_size)
print(ipc_data)
print(total_cycles)

df = pd.DataFrame(data)
#csv_file = '../dataset/hyperparameter_dataset/NN_model_batchsize.csv' # Specify your CSV file name
csv_file = "/home/pankhuri/PycharmProjects/ThesisProject/dataset/hyperparameter_dataset/NN_model_batchsize.csv"
df.to_csv(csv_file, index=False, mode = 'w')

plot_3_graphs(batch_size, ipc_data, total_cycles, accuracies)


