from tensorflow.keras.regularizers import l1, l2, l1_l2
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
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
def reset_random_seeds():
   np.random.seed(42)
   python_random.seed(42)
   tf.random.set_seed(42)

reset_random_seeds()

def plot_graph(dropout_rates, ipc, cycles):
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel("Dropout Rate")
    ax1.set_ylabel("IPC", color=color)
    ax1.plot(dropout_rates, ipc, color=color, marker='o')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel("accuracy", color=color)
    ax2.plot(dropout_rates, cycles, color=color, marker='x')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('IPC and accuracy vs. Dropout Rates')
    ax1.grid(True)
    plt.show()

def plot_3_graphs(regularization_types, ipc_data, total_cycles, accuracies):
    # Create 1x3 subplots
    fig, axs = plt.subplots(1, 3, figsize=(10, 5))

    # Plot IPC
    axs[0].plot(regularization_types, ipc_data, marker='o', color='red')
    axs[0].set_title('IPC vs Dropout Rates')
    axs[0].set_xlabel('Dropout Rate')
    axs[0].set_ylabel('IPC')

    # Plot Cycles
    axs[1].plot(regularization_types, total_cycles, marker='o', color='blue')
    axs[1].set_title('Cycles vs Dropout Rates')
    axs[1].set_xlabel('Dropout Rate')
    axs[1].set_ylabel('Cycles')

    # Plot Accuracy
    axs[2].plot(regularization_types, accuracies, marker='o', color='green')
    axs[2].set_title('Accuracy vs Dropout Rates')
    axs[2].set_xlabel('Dropout Rate')
    axs[2].set_ylabel('Accuracy')

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


def create_model(regularization_type, regularization_coefficient):
    if regularization_type == 'l1':
        regularizer = l1(regularization_coefficient)
    elif regularization_type == 'l2':
        regularizer = l2(regularization_coefficient)
    elif regularization_type == 'l1_l2':
        regularizer = l1_l2(l1=regularization_coefficient, l2=regularization_coefficient)
    else:
        regularizer = None

    model = Sequential([
        Dense(16, activation='relu', input_shape=(X_train_scaled.shape[1],), kernel_regularizer=regularizer),
        Dropout(0.5),
        Dense(16, activation='relu', kernel_regularizer=regularizer),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

regularization_types = ['none', 'l1', 'l2', 'l1_l2']
regularization_coefficients = [0.001]  # Example coefficients for regularization
ipc_data = []
total_cycles = []
accuracies = []
data = {}

for regularization_type in regularization_types:
    for coeff in regularization_coefficients:
        # Skip the 'none' type for non-zero coefficients
        if regularization_type == 'none' and coeff != regularization_coefficients[0]:
            continue

        print(f"Training with {regularization_type} regularization and coefficient {coeff}")
        model = create_model(regularization_type if regularization_type != 'none' else None, coeff)

        papi_high.start_counters([papi_events.PAPI_TOT_INS, papi_events.PAPI_TOT_CYC])

        # Train the model
        model.fit(X_train_scaled, y_train, epochs=100, batch_size=10, verbose=0)

        counters = papi_high.stop_counters()
        ins = counters[0]
        cycle = counters[1]
        ipc = ins / cycle if cycle > 0 else 0
        ipc_data.append(ipc)
        total_cycles.append(cycle)

        # Evaluate the model's accuracy on the test set
        loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
        accuracies.append(accuracy)

        print(f"Regularization: {regularization_type}, Coefficient: {coeff}, Test Accuracy: {accuracy:.4f}")

data["regularization type"] = [f"{rtype}_{coeff}" for rtype in regularization_types for coeff in regularization_coefficients if not (rtype == 'none' and coeff != regularization_coefficients[0])]
data["ipc"] = ipc_data
data["cycles"] = total_cycles
data["accuracies"] = accuracies

df = pd.DataFrame(data)
#csv_file = '../dataset/hyperparameter_dataset/NN_model_batchsize.csv' # Specify your CSV file name
csv_file = "/home/pankhuri/PycharmProjects/ThesisProject/dataset/hyperparameter_dataset/NN_model_regularisation.csv"
df.to_csv(csv_file, index=False, mode = 'w')

plot_3_graphs(regularization_types, ipc_data, total_cycles, accuracies)
