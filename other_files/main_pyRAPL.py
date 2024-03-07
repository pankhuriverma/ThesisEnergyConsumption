import pyRAPL
import numpy as np
import matplotlib.pyplot as plt
import csv
import tensorflow as tf
import time

pyRAPL.setup()


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
def measure_energy(epochs_num):
    x_train, y_train, model, loss_fn = model_initialise()
    pyrapl_output =[]
    pyRAPL.setup()
    meter = pyRAPL.Measurement('NN Model')
    start_time = time.time()
    meter.begin()
    train_model(epochs_num, x_train, y_train, model, loss_fn)
    meter.end()
    end_time = time.time()
    output = meter.result
    cpu_energy = output.pkg
    dram_energy = output.dram
    total_energy = cpu_energy[0] + dram_energy[0]
    total_time = end_time - start_time

    return cpu_energy[0]


if __name__ == "__main__":

    energy = []
    cpu_ener = []
    dram_ener = []
    tot_time = []
    num_of_epochs = list(range(1,2))
    print(len(num_of_epochs))
    x_train, y_train, model, loss_fn = model_initialise()
    for ne in num_of_epochs:
        print(ne)
        cpu_energy = measure_energy(ne)

        cpu_ener.append(cpu_energy)


    print("CPU Energy: ", sum(cpu_ener))


    """with open('piRAPL_output.csv', 'w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow(['No. of epochs','Total Energy'])
        for epoch, ener in zip(num_of_epochs, energy):
            writer.writerow([epoch, ener])


    plt.figure(figsize=(10, 6))
    plt.plot(num_of_epochs, energy, label='total energy')

    # Adding title and labels
    plt.title('num of epochs vs total energy')
    plt.xlabel('num of epochs')
    plt.ylabel('total energy')

    # Adding a grid for better readability
    plt.grid(True)

    # Showing the plot
    plt.legend()
    plt.show()"""