from pypapi import papi_high
from pypapi import events as papi_events
import numpy as np
import matplotlib.pyplot as plt
import csv
import tensorflow as tf
import statistics
import time



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
    start_time = time.time()
    papi_high.start_counters([papi_events.PAPI_TOT_INS, papi_events.PAPI_TOT_CYC])

    train_model(epochs_num, x_train, y_train, model, loss_fn)
    counters = papi_high.stop_counters()
    end_time = time.time()
    total_time = end_time - start_time
    return counters[0], counters[1], total_time



if __name__ == "__main__":


    tot_ins = []
    tot_cyc = []
    tot_time = []

    num_of_epochs = list(range(2,4))
    print(len(num_of_epochs))
    x_train, y_train, model, loss_fn = model_initialise()

    for ne in num_of_epochs:
        print(ne)
        ins, cyc, total_time = measure_performance(ne)
        tot_ins.append(ins)
        tot_cyc.append(cyc)
        tot_time.append(total_time)

    print('Total Instructions: ', sum(tot_ins))
    print('Total cycles: ', sum(tot_cyc))
    print('Total time: ', sum(tot_time))

    #req_exc_sha_cache_line, req_exc_cln_cache_line, data_trans_ls_buff_miss, ins_trans_ls_buff_misses, L1_load_miss, L1_store_miss = measure_performance(ne)
    """req_exc_acc_to_shared_cache_line.append(req_exc_sha_cache_line)
        req_exc_acc_to_clean_cache_line.append(req_exc_cln_cache_line)
        data_trans_lookaside_buff_misses.append(data_trans_ls_buff_miss)
        ins_trans_lookaside_buff_misses.append(ins_trans_ls_buff_misses)
        L1_load_misses.append(L1_load_miss)
        L1_store_misses.append(L1_store_miss)

    dependent_variables = [req_exc_acc_to_shared_cache_line, req_exc_acc_to_clean_cache_line, data_trans_lookaside_buff_misses, ins_trans_lookaside_buff_misses, L1_load_misses, L1_store_misses]
    dependent_variables_names = ['req_exc_acc_to_shared_cache_line', 'req_exc_acc_to_clean_cache_line', 'data_trans_lookaside_buff_misses', 'ins_trans_lookaside_buff_misses', 'L1_load_misses', 'L1_store_misses']


    with open('piPAPI_NN_output_2.csv', 'w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow(['No of Epochs', 'req_exc_acc_to_shared_cache_line', 'req_exc_acc_to_clean_cache_line', 'data_trans_lookaside_buff_misses', 'ins_trans_lookaside_buff_misses', 'L1_load_misses', 'L1_store_misses'])
        for epoch, cyc, ins, l1_dc_m, l1_ic_m, l1_c_m, rfs in zip(num_of_epochs, req_exc_acc_to_shared_cache_line, req_exc_acc_to_clean_cache_line, data_trans_lookaside_buff_misses, ins_trans_lookaside_buff_misses, L1_load_misses, L1_store_misses):
            writer.writerow([epoch, cyc, ins, l1_dc_m, l1_ic_m, l1_c_m, rfs])





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
    plt.show()"""

