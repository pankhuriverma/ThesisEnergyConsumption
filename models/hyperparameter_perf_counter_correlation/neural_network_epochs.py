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
import pyRAPL
pyRAPL.setup()
def reset_random_seeds():
   np.random.seed(42)
   python_random.seed(42)
   tf.random.set_seed(42)

reset_random_seeds()



def plot_3_graphs(epoch_list, total_instructions, total_cycles, cpu_energy, dram_energy, accuracies):
    # Create 1x3 subplots
    fig, axs = plt.subplots(2, 3, figsize=(10, 5))

    # Plot IPC
    axs[0,0].plot(epoch_list, total_instructions, marker='o', color='red')
    axs[0,0].set_title('Instructions vs Epochs')
    axs[0,0].set_xlabel('Epochs')
    axs[0,0].set_ylabel('Instructions')

    # Plot Cycles
    axs[0,1].plot(epoch_list, total_cycles, marker='o', color='blue')
    axs[0,1].set_title('Cycles vs Epochs')
    axs[0,1].set_xlabel('Epochs')
    axs[0,1].set_ylabel('Cycles')

    # Plot Accuracy
    axs[0,2].plot(epoch_list, accuracies, marker='o', color='green')
    axs[0,2].set_title('Accuracy vs Epochs')
    axs[0,2].set_xlabel('Epochs')
    axs[0,2].set_ylabel('Accuracy')

    # Plot CPU Energy
    axs[1, 0].plot(epoch_list, cpu_energy, marker='o', color='red')
    axs[1, 0].set_title('CPU Energy vs Epochs')
    axs[1, 0].set_xlabel('Epochs')
    axs[1, 0].set_ylabel('CPU Energy')

    # Plot DRAM Energy
    axs[1, 1].plot(epoch_list, dram_energy, marker='o', color='blue')
    axs[1, 1].set_title('DRAM Energy vs Epochs')
    axs[1, 1].set_xlabel('Epochs')
    axs[1, 1].set_ylabel('DRAM Energy')

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
def create_model():
    model = Sequential([
        Dense(16, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model



# Initialize a list to store the accuracy of each model
accuracies = []
total_instructions = []
total_cycles = []
cpu_energy = []
dram_energy = []
epoch_list = []
data = {}

# Loop over Epochss
for ep in range(50,29,-1):
    model = create_model()
    epoch_list.append(ep)
    meter = pyRAPL.Measurement('LR Model')
    meter.begin()
    papi_high.start_counters([papi_events.PAPI_TOT_INS, papi_events.PAPI_TOT_CYC])

    # Train the model
    model.fit(X_train_scaled, y_train, epochs=ep, batch_size=30, verbose=0)

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

    # Evaluate the model on the test set
    loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    accuracies.append(accuracy)
    print(f"Epochs: {ep}, Test Accuracy: {accuracy:.4f}")

data["Epochs"] = epoch_list
data["ins"] = total_instructions
data["cycles"] = total_cycles
data["accuracy"] = accuracies
data["cpu energy"] = cpu_energy
data["dram energy"] = dram_energy

print(epoch_list)
print(total_instructions)
print(total_cycles)

df = pd.DataFrame(data)
csv_file = "/home/pankhuri/PycharmProjects/ThesisProject/dataset/hyperparameter_dataset/NN_model_epochs.csv"
df.to_csv(csv_file, index=False, mode = 'w')

plot_3_graphs(epoch_list, total_instructions, total_cycles, cpu_energy, dram_energy, accuracies)
