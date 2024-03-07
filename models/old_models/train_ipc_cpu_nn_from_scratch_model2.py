import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def plot_graph(X, y1, y2):

    # Create a figure and a set of subplots
    fig, ax1 = plt.subplots()

    # Plot the first set of data and set axis labels
    color = 'tab:red'
    ax1.set_xlabel('X axis label')
    ax1.set_ylabel('Y1 axis label', color=color)
    ax1.scatter(X, y1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # Instantiate a second y-axis sharing the same x-axis
    ax2 = ax1.twinx()

    # Plot the second set of data with a different color
    color = 'tab:blue'
    ax2.set_ylabel('Y2 axis label', color=color)
    ax2.scatter(X, y2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # Title and grid
    plt.title('Graph with two Y values and one X value')
    ax1.grid(True)

    # Show the plot
    plt.show()




# Define the ReLU activation function and its derivative
def relu(Z):
    return np.maximum(0, Z)


def relu_derivative(Z):
    return Z > 0


# Initialize the parameters of the neural network
def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(2)
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }
    return parameters


# Forward propagation
def forward_propagation(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    Z1 = np.dot(W1, X.T) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = Z2

    cache = {
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2
    }

    return A2, cache


# Compute the Mean Absolute Error cost
def compute_cost(A2, Y):
    m = Y.shape[0]
    cost = (1. / (2*m)) * np.sum(np.square(A2 - Y))
    return cost


# Backward propagation
def backward_propagation(parameters, cache, X, Y):
    m = X.shape[0]

    W1 = parameters['W1']
    W2 = parameters['W2']

    A1 = cache['A1']
    A2 = cache['A2']

    dZ2 = A2 - Y
    dW2 = (1. / m) * np.dot(dZ2, A1.T)
    db2 = (1. / m) * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, relu_derivative(cache['Z1']))
    dW1 = (1. / m) * np.dot(dZ1, X)
    db1 = (1. / m) * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2
    }

    return gradients


# Update the parameters using gradient descent
def update_parameters(parameters, grads, learning_rate):
    parameters['W1'] -= learning_rate * grads['dW1']
    parameters['b1'] -= learning_rate * grads['db1']
    parameters['W2'] -= learning_rate * grads['dW2']
    parameters['b2'] -= learning_rate * grads['db2']

    return parameters


# Model training
def model(X_train, Y_train, n_h, num_iterations, learning_rate):
    n_x = X_train.shape[1]
    n_y = Y_train.shape[1]

    parameters = initialize_parameters(n_x, n_h, n_y)

    for i in range(0, num_iterations):
        A2, cache = forward_propagation(X_train, parameters)
        cost = compute_cost(A2, Y_train.T)
        grads = backward_propagation(parameters, cache, X_train, Y_train.T)
        parameters = update_parameters(parameters, grads, learning_rate)

        print(f"Cost after iteration {i}: {cost}")

    return parameters


# Predict function
def predict(parameters, X):
    A2, cache = forward_propagation(X, parameters)
    return A2


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
    X, y = clean_data(data)



    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=42)
    # Normalizing the data
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Assuming X_train, y_train, X_test, y_test are already defined and normalized
    # Train the model
    trained_parameters = model(X_train_scaled, y_train, n_h=100, num_iterations=100, learning_rate=0.001)

    # Make predictions on the test set
    y_pred_test = predict(trained_parameters, X_test_scaled)

    print(y_pred_test)


    plot_graph(X_test_scaled,y_test,y_pred_test)