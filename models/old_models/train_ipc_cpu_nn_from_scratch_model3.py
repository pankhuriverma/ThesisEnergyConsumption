import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


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

def plot_graph_2d(X, y):
    plt.figure(figsize=(10, 10))
    plt.scatter(X, y, label='graph')

    # Adding title and labels
    plt.title('X vs Y')
    plt.xlabel('X')
    plt.ylabel('Y')

    # Adding a grid for better readability
    plt.grid(True)


    # Showing the plot
    plt.legend()
    plt.show()


def backward_propagation(x, y, cache, parameters, m):

    A1, A2, A3 = cache["A1"], cache["A2"], cache["A3"]
    W2, W3 = parameters["W2"], parameters["W3"]

    dZ3 = A3 - y.T
    dW3 = (1 / m) * np.dot(dZ3, A2.T)
    db3 = (1 / m) * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, relu_derivative(cache["Z2"]))
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, relu_derivative(cache["Z1"]))
    dW1 = (1 / m) * np.dot(dZ1, x)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2, "dW3": dW3, "db3": db3}
    return gradients


def initialize_parameters(n_x, n_h1, n_h2, n_y):
    np.random.seed(42)
    W1 = np.random.randn(n_h1, n_x) * 0.01
    b1 = np.zeros((n_h1, 1))
    W2 = np.random.randn(n_h2, n_h1) * 0.01
    b2 = np.zeros((n_h2, 1))
    W3 = np.random.randn(n_y, n_h2) * 0.01
    b3 = np.zeros((n_y, 1))
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}
    print("W1, b1, W2, b2, W3, b3")
    print(W1.shape,b1.shape,W2.shape,b2.shape,W3.shape,b3.shape)

    return parameters
def update_parameters(parameters, grads, learning_rate):
    parameters["W1"] -= learning_rate * grads["dW1"]
    parameters["b1"] -= learning_rate * grads["db1"]
    parameters["W2"] -= learning_rate * grads["dW2"]
    parameters["b2"] -= learning_rate * grads["db2"]
    parameters["W3"] -= learning_rate * grads["dW3"]
    parameters["b3"] -= learning_rate * grads["db3"]
    return parameters

def relu(Z):
    return np.maximum(0, Z)

def relu_derivative(Z):
    return Z > 0
def forward_propagation(x, parameters):
    W1, b1 = parameters["W1"], parameters["b1"]
    W2, b2 = parameters["W2"], parameters["b2"]
    W3, b3 = parameters["W3"], parameters["b3"]

    Z1 = np.dot(W1, x.T) + b1
    print("!!!!!!")
    print(Z1.shape)
    A1 = relu(Z1)
    print(A1.shape)
    Z2 = np.dot(W2, A1) + b2
    print(Z2.shape)
    A2 = relu(Z2)
    print(A2.shape)
    Z3 = np.dot(W3, A2) + b3
    print(Z3.shape)
    A3 = Z3  # For regression, no activation function in the output layer
    print(A3.shape)
    print("!!!!!!")

    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2, "Z3": Z3, "A3": A3}
    return A3, cache
def compute_cost(A3, y, m):

    cost = (1 / (2*m)) * np.sum(np.square(A3 - y.T))
    return cost



# Model training for two hidden layers
def model(x, y, n_h1, n_h2, num_iterations, learning_rate):

    n_x = x.shape[1]
    print("n_x: ",n_x)
    n_y = y.shape[1]
    print("n_y: ",n_y)
    print("n_h1: ", n_h1)
    print("n_h2: ", n_h2)
    parameters = initialize_parameters(n_x, n_h1, n_h2, n_y)
    m = x.shape[0]
    print("m:", m)
    # The loop for forward propagation, cost computation, backward propagation, and parameter update

    for i in range(num_iterations):
        # Forward propagation
        A3, caches = forward_propagation(x, parameters)
        print("A3:", A3.shape)

        # Cost function
        cost = compute_cost(A3, y, m)

        # Backward propagation
        grads = backward_propagation(x, y, caches, parameters, m)

        # Update parameters
        parameters = update_parameters(parameters, grads, learning_rate)


        print(f"Cost after iteration {i}: {cost}")

    return A3, parameters



def predict(parameters, x):
    A3, cache = forward_propagation(x, parameters)
    return A3


# Data cleaning function
def clean_data(data):

    mean = data['cpu energy'].mean()
    std_dev = data['cpu energy'].std()
    threshold = 3  # 3 standard deviations
    data['Z_score'] = (data['cpu energy'] - mean) / std_dev
    cleaned_data = data[np.abs(data['Z_score']) <= threshold].drop(columns=['Z_score'])
    cleaned_data[['ipc']] = cleaned_data[['ipc']].round(4)
    cleaned_data[["cpu energy"]] = cleaned_data[["cpu energy"]].round(4)
    return cleaned_data[['ipc']], cleaned_data[["cpu energy"]]

if __name__ == "__main__":
    # Load the dataset
    csv_file_path = '../../dataset/ipc_dataset/ML_model_collected_dataset_ipc_10iterations_avg.csv'  # Change this to the correct path
    data = pd.read_csv(csv_file_path)
    #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


    X, y = clean_data(data)

    """X = data[["ipc"]]
    y = data[["cpu energy"]]"""

    X_train, X_test, Y_train, Y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=42)
    print("X_train, Xtest, y_train, y_test")
    print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)
    plot_graph_2d(X_train, Y_train)
    A3, parameters = model(X_train, Y_train, n_h1=100, n_h2=100, num_iterations=100, learning_rate=0.001)
    y_pred_test = predict(parameters, X_test)
    plot_graph(X_train.flatten(), Y_train.flatten(), A3.flatten())
    print(y_pred_test)
