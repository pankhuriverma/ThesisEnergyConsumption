import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
def test_model(X_test, y_test, parameters):

    # Perform forward propagation with the trained parameters
    y_test_pred, _ = forward_propagation(X_test, parameters)

    # Calculate the MSE cost
    cost = compute_cost(y_test_pred, y_test.reshape(-1, 1))

    return cost, y_test_pred


def graph_plot(costs):

    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('epochs (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

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




def relu(Z):
    return np.maximum(0, Z)


def relu_derivative(Z):
    return Z > 0


def forward_propagation(X, parameters):
    # Retrieve parameters
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    # First hidden layer
    Z1 = np.dot(W1, X.T) + b1
    A1 = relu(Z1)

    # Second hidden layer
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)

    # Output layer
    Z3 = np.dot(W3, A2) + b3
    A3 = Z3  # No activation function for the output layer in regression

    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2, "Z3": Z3, "A3": A3}
    return A3, cache


def compute_cost(A3, Y):
    m = Y.shape[0]
    cost = (1 / (2*m)) * np.sum(np.square(A3 - Y.T))
    return cost


def backward_propagation(X, Y, cache, parameters):
    m = X.shape[0]
    # Retrieve parameters
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]
    # Retrieve forward propagation results
    A1 = cache["A1"]
    A2 = cache["A2"]
    A3 = cache["A3"]

    # Output layer gradients
    dZ3 = A3 - Y.T
    dW3 = (1 / m) * np.dot(dZ3, A2.T)
    db3 = (1 / m) * np.sum(dZ3, axis=1, keepdims=True)

    # Second hidden layer gradients
    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, relu_derivative(cache["Z2"]))
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    # First hidden layer gradients
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, relu_derivative(cache["Z1"]))
    dW1 = (1 / m) * np.dot(dZ1, X)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2, "dW3": dW3, "db3": db3}
    return gradients


def update_parameters(parameters, gradients, learning_rate):
    parameters["W1"] -= learning_rate * gradients["dW1"]
    parameters["b1"] -= learning_rate * gradients["db1"]
    parameters["W2"] -= learning_rate * gradients["dW2"]
    parameters["b2"] -= learning_rate * gradients["db2"]
    parameters["W3"] -= learning_rate * gradients["dW3"]
    parameters["b3"] -= learning_rate * gradients["db3"]

    return parameters

def clean_data(data):

    mean = data['cpu energy'].mean()
    std_dev = data['cpu energy'].std()

    # Set the threshold (e.g., 3 standard deviations)
    threshold = 3

    # Calculate Z-score for each data point
    data['Z_score'] = (data['cpu energy'] - mean) / std_dev

    # Identify outliers based on the threshold
    outliers = data[np.abs(data['Z_score']) > threshold]

    # Remove rows with outliers
    cleaned_data = data[np.abs(data['Z_score']) <= threshold]

    # Optionally, remove the 'Z_score' column from the cleaned data
    cleaned_data = cleaned_data.drop(columns=['Z_score'])
    #print(cleaned_data)
    cleaned_data[['ipc']] = cleaned_data[['ipc']].round(4)

    X = cleaned_data[['ipc']]

    print(X)
    # Dependent variable



    cleaned_data[["cpu energy"]] = cleaned_data[["cpu energy"]].round(4)

    y = cleaned_data[["cpu energy"]]


    return X, y



if __name__ == "__main__":
    # Re-import necessary libraries


    # Load the dataset
    csv_file_path = '../../dataset/ipc_dataset/ML_model_collected_dataset_ipc_10iterations_avg.csv'
    data = pd.read_csv(csv_file_path)

    # Selecting the independent variable (X) and the dependent variable (y)
    """X = data[['ipc']].values  # Independent variable
    y = data[['cpu energy']].values """ # Dependent variable
    X, y = clean_data(data)

    #X, y = clean_data(data)
    # Normalize the data
    """    X_mean, X_std = X.mean(), X.std()
    y_mean, y_std = y.mean(), y.std()

    X_norm = (X - X_mean) / X_std
    y_norm = (y - y_mean) / y_std"""

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=42)

    # Verify the shapes of the splits
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # Neural Network architecture parameters
    n_x = X_train.shape[1]  # Number of input features
    n_h1 = 100 # Number of neurons in first hidden layer
    n_h2 = 100  # Number of neurons in second hidden layer
    n_y = y_train.shape[1]  # Number of output neurons

    # Initialize parameters
    np.random.seed(1)  # Seed the random number generator for reproducibility
    W1 = np.random.randn(n_h1, n_x) * 0.01
    b1 = np.zeros((n_h1, 1))
    W2 = np.random.randn(n_h2, n_h1) * 0.01
    b2 = np.zeros((n_h2, 1))
    W3 = np.random.randn(n_y, n_h2) * 0.01
    b3 = np.zeros((n_y, 1))

    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}

    # Training loop
    learning_rate = 0.001
    num_epochs = 100
    costs = []

    for epoch in range(num_epochs):
        A3, cache = forward_propagation(X_train, parameters)
        cost = compute_cost(A3, y_train)
        gradients = backward_propagation(X_train, y_train, cache, parameters)
        parameters = update_parameters(parameters, gradients, learning_rate)

        # Print the cost every 10 epochs

        print(f"Cost after epoch {epoch}: {cost}")
        costs.append(cost)



    # To make predictions, use the forward_propagation function with X_test and the trained parameters
    A3_test, _ = forward_propagation(X_test, parameters)
    #y_pred_test = A3_test.T * y_std + y_mean  # Un-normalize predictions




    y_pred_list = []
    for i in range(len(A3_test)):
        y_pred_list.append(A3_test[i])


    y_test_list = []
    for i in range(len(y_test)):
        y_test_list.append(y_test[i][0])


    print(y_pred_list)
    print("!!!!!!!")
    print(y_test_list)
    plot_graph(X_test, y_test_list, y_pred_list)
    to_csvfile = {}

    to_csvfile["pred"] = y_pred_list
    to_csvfile["true"] = y_test

    df = pd.DataFrame(to_csvfile)
    csv_file = '../dataset/ipc_dataset/NN_model_test_compare_avg1.csv'  # Specify your CSV file name
    df.to_csv(csv_file, index=False, mode='w')