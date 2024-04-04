"""import tensorflow as tf
print("GPUs Available: ", tf.config.list_physical_devices('GPU'))


print(tf.test.is_built_with_cuda())
# Print TensorFlow build information
print(tf.sysconfig.get_build_info())
print("XLA version:", tf.sysconfig.get_compile_flags()[0])"""
"""import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import subprocess

# Define a simple CNN model
def create_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    return model

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Preprocess the data
train_images = train_images.reshape((60000, 28, 28, 1)) / 255.0
test_images = test_images.reshape((10000, 28, 28, 1)) / 255.0
train_labels = tf.keras.utils.to_categorical(train_labels, 10)
test_labels = tf.keras.utils.to_categorical(test_labels, 10)

# Create the model
model = create_model()

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=5, batch_size=64)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

# Save the model
model.save('mnist_cnn_model.h5')

# Use nvprof to measure performance counters
def measure_performance_counters():
    try:
        # Execute the command to measure performance counters using nvprof
        result = subprocess.run(["nvprof", "--print-gpu-trace", "python", "new.py"],
                                capture_output=True, text=True)

        # Print the output of nvprof
        print(result.stdout)

    except Exception as e:
        print("An error occurred while measuring performance counters with nvprof:", e)

# Call the function to measure performance counters
measure_performance_counters()
"""

import cupy as cp

# Generate synthetic data
n_samples = 1000
n_features = 10
n_classes = 2

X = cp.random.randn(n_samples, n_features)
y = cp.random.randint(n_classes, size=n_samples)

# Define neural network architecture
n_hidden = 64

# Initialize weights and biases
W1 = cp.random.randn(n_features, n_hidden)
b1 = cp.zeros((1, n_hidden))
W2 = cp.random.randn(n_hidden, n_classes)
b2 = cp.zeros((1, n_classes))


# Define activation function (e.g., ReLU)
def relu(x):
    return cp.maximum(0, x)


# Define softmax function for output layer
def softmax(x):
    exp_x = cp.exp(x - cp.max(x, axis=1, keepdims=True))
    return exp_x / cp.sum(exp_x, axis=1, keepdims=True)


# Define forward pass function
def forward_pass(X, W1, b1, W2, b2):
    z1 = cp.dot(X, W1) + b1
    a1 = relu(z1)
    z2 = cp.dot(a1, W2) + b2
    y_pred = softmax(z2)
    return y_pred


# Define cross-entropy loss function
def cross_entropy_loss(y_true, y_pred):
    return -cp.mean(cp.log(y_pred[cp.arange(len(y_true)), y_true]))


# Define learning rate
learning_rate = 0.01

# Training loop
for epoch in range(100):
    # Forward pass
    y_pred = forward_pass(X, W1, b1, W2, b2)

    # Compute loss
    loss = cross_entropy_loss(y, y_pred)

    # Print loss every few epochs
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss}')

    # Backpropagation
    grad_y_pred = cp.copy(y_pred)
    grad_y_pred[cp.arange(len(y)), y] -= 1
    grad_W2 = cp.dot(relu(cp.dot(X, W1) + b1).T, grad_y_pred)
    grad_b2 = cp.sum(grad_y_pred, axis=0, keepdims=True)
    grad_W1 = cp.dot(X.T, cp.dot(grad_y_pred, W2.T) * (cp.dot(X, W1) + b1 > 0))
    grad_b1 = cp.sum(cp.dot(grad_y_pred, W2.T) * (cp.dot(X, W1) + b1 > 0), axis=0, keepdims=True)

    # Update weights and biases
    W1 -= learning_rate * grad_W1
    b1 -= learning_rate * grad_b1
    W2 -= learning_rate * grad_W2
    b2 -= learning_rate * grad_b2

# Perform inference
y_pred = forward_pass(X, W1, b1, W2, b2)
predictions = cp.argmax(y_pred, axis=1)
accuracy = cp.mean(predictions == y)
print(f'Accuracy: {accuracy}')
