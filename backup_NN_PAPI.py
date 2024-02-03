from pypapi import papi_high
from pypapi import events as papi_events
import numpy as np
import matplotlib.pyplot as plt
import csv
import tensorflow as tf


def model_initialise():
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
    return x_train, y_train, model, loss_fn

def train_model(epochs_num, x_train, y_train, model, loss_fn):

    model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=epochs_num)

    return model


def measure_performance(epochs_num):
    papi_output = []
    x_train, y_train, model, loss_fn = model_initialise()
    papi_high.start_counters([papi_events.PAPI_TOT_INS, papi_events.PAPI_TOT_CYC, papi_events.PAPI_L1_DCM,
                              papi_events.PAPI_L1_ICM, papi_events.PAPI_L1_TCM, papi_events.PAPI_CA_SNP])
    train_model(epochs_num, x_train, y_train, model, loss_fn)
    counters = papi_high.stop_counters()
    return counters[0], counters[1], counters[2], counters[3], counters[4], counters[5]


if __name__ == "__main__":

    total_cycles = []
    total_instructions = []
    L1_data_cache_misses = []
    L1_ins_cache_misses = []
    L1_cache_misses = []
    req_for_snoops = []

    num_of_epochs = list(range(1, 10))
    print(len(num_of_epochs))
    x_train, y_train, model, loss_fn = model_initialise()

    for ne in num_of_epochs:
        print(ne)
        instructions, cycles, L1_data_cache_miss, L1_ins_cache_miss, L1_cache_miss, req_for_snoop = measure_performance(ne)
        total_cycles.append(cycles)
        total_instructions.append(instructions)
        L1_data_cache_misses.append(L1_data_cache_miss)
        L1_ins_cache_misses.append(L1_ins_cache_miss)
        L1_cache_misses.append(L1_cache_miss)
        req_for_snoops.append(req_for_snoop)

    dependent_variables = [total_instructions, total_cycles, L1_data_cache_misses, L1_ins_cache_misses, L1_cache_misses, req_for_snoops]
    dependent_variables_names = ['total_instructions', 'total_cycles', 'L1_data_cache_misses', 'L1_ins_cache_misses', 'L1_cache_misses', 'req_for_snoops']


    with open('piPAPI_NN_output_1.csv', 'w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow(['No of Epochs', 'Cycles', 'Instructions', 'L1 data cache misses', 'L1 ins cache misses', 'L1 cache misses', 'Request for snoops'])
        for epoch, cyc, ins, l1_dc_m, l1_ic_m, l1_c_m, rfs in zip(num_of_epochs, total_cycles, total_instructions, L1_data_cache_misses, L1_ins_cache_misses, L1_cache_misses, req_for_snoops):
            writer.writerow([epoch, cyc, ins, l1_dc_m, l1_ic_m, l1_c_m, rfs])

    """plt.figure(figsize=(10, 6))
    plt.plot(num_of_epochs, total_cycles, label='cycles')

    # Adding title and labels
    plt.title('Matrix Size vs Cycles')
    plt.xlabel('Epochs')
    plt.ylabel('Cycles')

    # Adding a grid for better readability
    plt.grid(True)

    # Showing the plot
    plt.legend()
    plt.show()"""

    import matplotlib.pyplot as plt

    # Assuming you have the data for num_of_epochs and dependent_variables already defined

    # Create a 2x3 grid of subplots
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    # Set a title for the entire figure
    fig.suptitle('Epochs vs Performance Counters ', fontsize=16)

    # Loop to create each subplot
    for i in range(2):
        for j in range(3):
            axs[i, j].plot(num_of_epochs,
                           dependent_variables[i * 3 + j])  # Calculate the correct index for dependent_variables
            axs[i, j].set_ylabel(dependent_variables_names[i * 3 + j])  # Calculate the correct label
            axs[i, j].grid(True)

    # Label for the common x-axis
    axs[-1, 1].set_xlabel('No. of epochs')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the padding between and around subplots
    plt.show()

