from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

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
    #cleaned_data[['ipc']] = cleaned_data[['ipc']].round(4)

    X = cleaned_data[['ipc']]


    # Dependent variable



    #cleaned_data[["cpu energy"]] = cleaned_data[["cpu energy"]].round(4)

    y = cleaned_data[["cpu energy"]]


    return X, y

def plot_graph(X, y1, y2):

    # Create a figure and a set of subplots
    fig, ax1 = plt.subplots()

    # Plot the first set of data and set axis labels
    color = 'tab:red'
    ax1.set_xlabel('Instructions per Cycle (IPC)')
    ax1.set_ylabel('CPU Energy (Test)', color=color)
    ax1.scatter(X, y1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # Instantiate a second y-axis sharing the same x-axis
    ax2 = ax1.twinx()

    # Plot the second set of data with a different color
    color = 'tab:blue'
    ax2.set_ylabel('CPU Energy (Predicted)', color=color)
    ax2.scatter(X, y2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # Title and grid
    plt.title('Graph of IPC vs CPU Energy')
    ax1.grid(True)

    # Show the plot
    plt.show()

def plot_2d_graph(X,Y):
    plt.figure(figsize=(10, 10))
    plt.scatter(X, Y, label='graph')

    # Adding title and labels
    plt.title('CPU Energy (True vs Predicted)')
    plt.xlabel('CPU Energy (True)')
    plt.ylabel('CPU Energy (Predicted)')

    # Adding a grid for better readability
    plt.grid(True)


    # Showing the plot
    plt.legend()
    plt.show()

csv_file_path = '../dataset/ipc_dataset/ML_model_collected_dataset_ipc_10iterations_avg.csv'
data = pd.read_csv(csv_file_path)

X, y = clean_data(data)
print("!!! X values")
print(X.values)




"""# Selecting the independent variable (X) and the dependent variable (y)
X = data[['ipc']]  # Independent variable
y = data[['cpu energy']]  # Dependent variable"""

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=42)
print("!!! X train")
print(X_train)
print("!!! y train")
print(y_train)
# Normalizing the data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
y_train_scaled = scaler.fit_transform(y_train)
y_test_scaled = scaler.transform(y_test)
print("!!! X train scaled")
print((X_train_scaled))

# Verify the shapes of the splits
#print(X_train_scaled.shape, X_test_scaled.shape, y_train.shape, y_test.shape)

# Define the neural network model
np.random.seed(42)
model = Sequential([
    Dense(50, activation='relu', input_shape=(1,)),
    #Dense(20, activation='relu'),
    Dense(1)  # Output layer
])

model.compile(optimizer=Adam(learning_rate=0.0001), loss='huber')
history = model.fit(X_train_scaled, y_train_scaled, validation_split=0.2, epochs=200, batch_size=64, verbose=1)





layer1_weights = [model.layers[0].get_weights()]
print("layer1")
print(layer1_weights)
layer2_weights = [model.layers[1].get_weights()]
print("layer2")
print(layer2_weights)




test_loss = model.evaluate(X_test_scaled, y_test_scaled, verbose=1)
print(f"Test Loss: {test_loss}")

y_pred = model.predict(X_test_scaled)

X_test_unscaled = scaler.inverse_transform(X_test_scaled)
y_test_unscaled = scaler.inverse_transform(y_test_scaled)
y_pred_unscaled = scaler.inverse_transform(y_pred)

#Y_pred_scaled = scaler.transform(y_pred)
plot_graph(X_test_unscaled, y_test_unscaled, y_pred_unscaled)



y_pred_list = []
for i in range(len(y_pred_unscaled)):
    y_pred_list.append(y_pred_unscaled[i][0])

y_test_list = []
for i in range(len(y_test_unscaled)):
    y_test_list.append(y_test_unscaled[i][0])

to_csvfile = {}

plot_2d_graph(y_test_list,y_pred_list)

to_csvfile["pred"] = y_pred_list
to_csvfile["true"] = y_test_list

df = pd.DataFrame(to_csvfile)
csv_file = '../dataset/ipc_dataset/NN_model_ipc_cpu_pred_test_huber_loss_compare.csv' # Specify your CSV file name
df.to_csv(csv_file, index=False, mode = 'w')