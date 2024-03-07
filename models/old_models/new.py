import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd


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

 # Function to generate parameters of the linear regression model, m & b.
def init_params():
    np.random.seed(2)
    m = np.random.normal(scale=10)

    b = np.random.normal(scale=10)
    print("!!!!!")
    print(m)
    print("!!!!!")
    print(b)

    return m, b


def plot_graph(dataset, pred_line=None):
    X, y = dataset['X'], dataset['y']

    # Plot the set of datapoints
    _ = plt.scatter(X, y, alpha=0.8)

    if (pred_line != None):

        x_line, y_line = pred_line['x_line'], pred_line['y_line']

        _ = plt.plot(x_line, y_line, linewidth=2, markersize=12, color='red',
                     alpha=0.8)  # Plot the randomly generated line

        _ = plt.title('Random Line on set of Datapoints')

    else:
        _ = plt.title('Plot of Datapoints')

    _ = plt.xlabel('x')
    _ = plt.ylabel('y')

    plt.show()

# Function to plot predicted line
def plot_pred_line(X, y, m, b):
    # Generate a set of datapoints on x for creating a line.
    x_line = np.linspace(np.min(X), np.max(X), 10)

    # Calculate the corresponding y with random values of m & b
    y_line = m * x_line + b

    dataset = {'X': X, 'y': y}

    pred_line = {'x_line': x_line, 'y_line': y_line}

    plot_graph(dataset, pred_line)

    return


def forward_prop(X, m, b):
    y_pred = m * X + b

    return y_pred


def compute_loss(y, y_pred):
    loss = 1 / 2 * np.mean((y_pred - y) ** 2)

    return loss


def grad_desc(m, b, X_train, y_train, y_pred):
    dm = np.mean((y_pred - y_train) * X_train)
    db = np.mean(y_pred - y_train)

    return dm, db


def update_params(m, b, dm, db, l_r):
    m -= l_r * dm
    b -= l_r * db

    return m, b


def back_prop(X_train, y_train, y_pred, m, b, l_r):
    dm, db = grad_desc(m, b, X_train, y_train, y_pred)

    m, b = update_params(m, b, dm, db, l_r)

    return m, b

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
    #plot_graph_2d(X,y)
    """X = data[["ipc"]]
    y = data[["cpu energy"]]"""
    m, b = init_params()
    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=42)
    print("X_train, Xtest, y_train, y_test")
    print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
    epochs = 100
    losses =[]
    l_r = 0.05
    print(y_test.shape)
    print(X_test.shape)

    for i in range(epochs):
        y_pred = forward_prop(X_train, m, b)
        print(y_pred.shape)

        loss = compute_loss(y_train, y_pred)
        losses.append(loss)

        m, b = back_prop(X_train, y_train, y_pred, m, b, l_r)


        print('Epoch: ', i)
        print('Loss = ', loss)
            #plot_pred_line(X_train, y_train, m, b, losses)
    # Call function to generate paramets

    plot_graph(X_train,y_train,y_pred)

    print('Prediction: ')
    y_pred_test = forward_prop(X_test, m, b)
    plot_graph(X_test, y_test, y_pred_test)
    loss = compute_loss(y_test, y_pred_test)
    print('Loss = ', loss)
    accuracy = np.mean(np.fabs((y_pred_test - y_test) / y_test)) * 100
    print('Accuracy = {}%'.format(round(accuracy, 4)))


    print('Hence \nm = ', m)
    print('b = ', b)

    y_pred_list = []
    for i in range(len(y_pred_test)):
        y_pred_list.append(y_pred_test[i][0])

    y_test_list = []
    for i in range(len(y_test)):
        y_test_list.append(y_test[i][0])

    to_csvfile = {}

    to_csvfile["pred"] = y_pred_list
    to_csvfile["true"] = y_test_list

    df = pd.DataFrame(to_csvfile)
    csv_file = '../../dataset/ipc_dataset/old_dataset/NN_model_test_compare_avg_new.csv'  # Specify your CSV file name
    df.to_csv(csv_file, index=False, mode='w')





