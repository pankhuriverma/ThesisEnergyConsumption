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
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
def reset_random_seeds():
   np.random.seed(42)
   python_random.seed(42)
   tf.random.set_seed(42)

reset_random_seeds()

def plot_graph(lr, ipc, cycles):


    # Create a figure and a set of subplots
    fig, ax1 = plt.subplots()

    # Plot the first set of data and set axis labels
    color = 'tab:red'
    ax1.set_xlabel("layers")
    ax1.set_ylabel("ipc", color=color)
    ax1.plot(lr, ipc, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # Instantiate a second y-axis sharing the same x-axis
    ax2 = ax1.twinx()

    # Plot the second set of data with a different color
    color = 'tab:blue'
    ax2.set_ylabel("cycles", color=color)
    ax2.plot(lr, cycles, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # Title and grid
    plt.title('Graph of Peformance Counters vs  Energy (Huber Loss)')
    ax1.grid(True)

    # Show the plot
    plt.show()

def plot_3_graphs(lrate, ipc_data, total_cycles, accuracies):
    # Create 1x3 subplots
    fig, axs = plt.subplots(1, 3, figsize=(10, 5))

    # Plot IPC
    axs[0].plot(lrate, ipc_data, marker='o', color='red')
    axs[0].set_title('IPC vs Learning Rate')
    axs[0].set_xlabel('Learning Rate')
    axs[0].set_ylabel('IPC')

    # Plot Cycles
    axs[1].plot(lrate, total_cycles, marker='o', color='blue')
    axs[1].set_title('Cycles vs Learning Rate')
    axs[1].set_xlabel('Learning Rate')
    axs[1].set_ylabel('Cycles')

    # Plot Accuracy
    axs[2].plot(lrate, accuracies, marker='o', color='green')
    axs[2].set_title('Accuracy vs Learning Rate')
    axs[2].set_xlabel('Learning Rate')
    axs[2].set_ylabel('Accuracy')

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
def create_model(learning_rate):
    model = Sequential([
        Dense(16, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model


# List of learning rates to try
learning_rates = [0.01, 0.001, 0.0001, 0.00001]

# Initialize a list to store the accuracy of each model
accuracies = []
ipc_data = []
total_cycles = []
lrate = []
data = {}

# Loop over learning rates
for lr in learning_rates:
    model = create_model(learning_rate=lr)
    lrate.append(lr)
    papi_high.start_counters([papi_events.PAPI_TOT_INS, papi_events.PAPI_TOT_CYC])

    # Train the model
    model.fit(X_train_scaled, y_train, epochs=100, batch_size=30, verbose=0)

    counters = papi_high.stop_counters()

    ins = counters[0]
    cycle = counters[1]
    ipc = ins / cycle if cycle > 0 else 0
    ipc_data.append(ipc)
    total_cycles.append(cycle)
    # Evaluate the model on the test set
    loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    accuracies.append(accuracy)
    print(f"Learning Rate: {lr}, Test Accuracy: {accuracy:.4f}")

data["learning rate"] = lrate
data["ipc"] = ipc_data
data["cycles"] = total_cycles
data["accuracy"] = accuracies

print(lrate)
print(ipc_data)
print(total_cycles)

df = pd.DataFrame(data)
csv_file = "/home/pankhuri/PycharmProjects/ThesisProject/dataset/hyperparameter_dataset/NN_model_lr.csv"
df.to_csv(csv_file, index=False, mode = 'w')

plot_3_graphs(lrate, ipc_data, total_cycles, accuracies)
