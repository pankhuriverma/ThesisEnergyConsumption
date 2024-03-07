# Imports
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
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

def clean_data(data):

    mean = data['cpu energy'].mean()
    std_dev = data['cpu energy'].std()
    threshold = 3  # 3 standard deviations
    data['Z_score'] = (data['cpu energy'] - mean) / std_dev
    cleaned_data = data[np.abs(data['Z_score']) <= threshold].drop(columns=['Z_score'])
    cleaned_data[['ipc']] = cleaned_data[['ipc']].round(4)
    cleaned_data[["cpu energy"]] = cleaned_data[["cpu energy"]].round(4)
    return cleaned_data[['ipc']], cleaned_data[["cpu energy"]]
def relu(x):
    return (x > 0) * x


def relu_derivative(output):
    return output > 0

# choose Transition function
transition_func = relu
transition_func_derivative =relu_derivative

np.random.seed(seed=1)
# -- Params
# Step size
alpha = 0.01

# Number of iterations
iters = 10

# Number of nuerons
hidden_size = 20

np.random.seed(42)
w1 = np.random.randn(100, 1) * 0.01
w2 = np.random.randn(100, 100) * 0.01

csv_file_path = '../../dataset/ipc_dataset/ML_model_collected_dataset_ipc_10iterations_avg.csv'  # Change this to the correct path
data = pd.read_csv(csv_file_path)
X, y = clean_data(data)
X_train, X_test, Y_train, Y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=42)
print("X_train, Xtest, y_train, y_test")
print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)
plot_graph_2d(X_train, Y_train)



for iteration in range(iters):
    for i in range(len(X_train)):
        # batch
        layer_0 = X_train
        print(layer_0.shape)

        # - Forward -
        y_hat = transition_func(layer_0 @ w1.T) @ w2

        # - Backward -
        # Predicted vs actual
        diff = y_hat - Y_train

        # w2 gradient
        w2_grad = diff.T @ transition_func(layer_0 @ w1.T)

        # w1 gradient
        w1_grad = (diff @ w2.T * transition_func_derivative(layer_0 @ w1.T)).T @ layer_0

        # update weights
        w2 -= alpha * w2_grad.T
        w1 -= alpha * w1_grad

    if iteration % 50 == 0:
        y_pred = transition_func(X_train @ w1) @ w2
        n = len(X_train)
        e = sum((y_pred - y) ** 2 / n)[0]


        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])
        ax.scatter(x=X_train[:, 1], y=y, color='r')
        ax.scatter(X_train[:, 1], y_pred, color='darkblue')
        ax.text(X_train.min(), Y_train.min(), f'func: RELU\nrmse: {e:.3}\niteration: {iteration} ',
                bbox=dict(facecolor='darkblue', alpha=0.01))

        plt.show()




