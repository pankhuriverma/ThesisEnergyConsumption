from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

def initialize_parameters(n_x, n_h, n_y):

    np.random.seed(42)  # Ensure reproducibility

    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    params = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }

    return params

def relu(Z):
    return np.maximum(0, Z)

def relu_derivative(Z):
    return Z > 0


def forward_propagation(X, params):

    # Retrieve each parameter from the dictionary "parameters"
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    # Implement Forward Propagation to calculate A2 (probabilities)
    Z1 = np.dot(W1, X.T) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = Z2  # No activation function for the output layer in regression

    cache = {
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2,

    }

    return A2, cache


def compute_loss(A2, Y):

    m = Y.shape[1]  # number of examples

    # Compute the MAE
    loss = np.sum(np.abs(Y.T - A2)) / m

    return loss


def backward_propagation(params, cache, X, Y):

    m = X.shape[1]

    # First, retrieve W1 and W2 from the dictionary "parameters".
    W1 = params['W1']
    W2 = params['W2']

    # Retrieve also A1 and A2 from dictionary "cache".
    A1 = cache['A1']
    A2 = cache['A2']
    Z1 = cache['Z1']

    # Backward propagation: calculate dW1, db1, dW2, db2.
    dZ2 = A2 - Y.T
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    dZ1 = np.dot(W2.T, dZ2) * relu_derivative(A1)  # Corrected implementation
    dW1 = np.dot(dZ1, X) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    grads = {
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2
    }

    return grads


def update_parameters(params, grads, learning_rate):

    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    # Retrieve each gradient from the dictionary "grads"
    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']

    # Update rule for each parameter
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    params = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }

    return params

def plot_graph(x, y1, y2):

    # Create a figure and a set of subplots
    fig, ax1 = plt.subplots()

    # Plot the first set of data and set axis labels
    color = 'tab:red'
    ax1.set_xlabel('IPC')
    ax1.set_ylabel('Test CPU Energy', color=color)
    ax1.scatter(x, y1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # Instantiate a second y-axis sharing the same x-axis
    ax2 = ax1.twinx()

    # Plot the second set of data with a different color
    color = 'tab:blue'
    ax2.set_ylabel('Pred CPU Energy', color=color)
    ax2.scatter(x, y2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # Title and grid
    plt.title('Graph with two Y values and one X value')
    ax1.grid(True)

    # Show the plot
    plt.show()


# Re-loading the dataset in case the context was reset
data_path = '../../dataset/ipc_dataset/ML_model_collected_dataset_ipc_10iterations_avg.csv'
data = pd.read_csv(data_path)

# Select ipc as independent variable (X) and cpu energy as dependent variable (Y)
X = data['ipc'].values.reshape(-1, 1)
Y = data['cpu energy'].values.reshape(-1, 1)

# Normalize the data
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_x.fit_transform(X)
Y_scaled = scaler_y.fit_transform(Y)

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y_scaled, test_size=0.2, random_state=42)

# Check the shape of the training and testing sets
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)


# Initialize parameters for our network
n_x = X_train.shape[1]  # Number of input features
n_h = 100  # Number of hidden units
n_y = 1  # Number of output units (since we're doing regression, this is 1)

params = initialize_parameters(n_x, n_h, n_y)

epochs = 100  # Example epoch count
learning_rate = 0.01  # Example learning rate

for i in range(epochs):
    # Forward propagation
    A2, cache = forward_propagation(X_train, params)

    # Compute loss
    loss = compute_loss(A2, Y_train)

    # Backward propagation
    grads = backward_propagation(params, cache, X_train, Y_train)

    # Update parameters
    params = update_parameters(params, grads, learning_rate)

    if i % 100 == 0:  # Example: print every 100 epochs
        print(f"Loss after iteration {i}: {loss}")


plot_graph(X_train,Y_train, A2)
