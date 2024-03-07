from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

def clean_data(data, feature, target):

    mean = data[target].mean()
    std_dev = data[target].std()

    # Set the threshold (e.g., 3 standard deviations)
    threshold = 3

    # Calculate Z-score for each data point
    data['Z_score'] = (data[target] - mean) / std_dev

    # Identify outliers based on the threshold
    outliers = data[np.abs(data['Z_score']) > threshold]

    # Remove rows with outliers
    cleaned_data = data[np.abs(data['Z_score']) <= threshold]

    # Optionally, remove the 'Z_score' column from the cleaned data
    cleaned_data = cleaned_data.drop(columns=['Z_score'])

    #cleaned_data[['ipc']] = cleaned_data[['ipc']].round(4)

    #X = cleaned_data[[feature]]


    # Dependent variable



    #cleaned_data[["cpu energy"]] = cleaned_data[["cpu energy"]].round(4)

    #y = cleaned_data[[target]]

    return cleaned_data

def plot_graph(X, y1, y2, i, j,X_feature, y_target1, y_target2):
    x = X[:, i]
    y1 = y1[:, j]
    y2 = y2[:, j]

    # Create a figure and a set of subplots
    fig, ax1 = plt.subplots()

    # Plot the first set of data and set axis labels
    color = 'tab:red'
    ax1.set_xlabel(X_feature)
    ax1.set_ylabel(y_target1, color=color)
    ax1.scatter(x, y1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # Instantiate a second y-axis sharing the same x-axis
    ax2 = ax1.twinx()

    # Plot the second set of data with a different color
    color = 'tab:blue'
    ax2.set_ylabel(y_target2, color=color)
    ax2.scatter(x, y2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # Title and grid
    plt.title('Graph of Peformance Counters vs  Energy (Huber Loss)')
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

def plot_3d_graph(X_test_unscaled, y_test_unscaled, y_pred_unscaled):
    from mpl_toolkits.mplot3d import Axes3D

    # Assuming the same setup as before

    fig = plt.figure(figsize=(12, 6))

    # 3D plot for the first dependent variable
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(y_test_unscaled, y_pred_unscaled, color='blue', label='Actual vs Predicted y1')
    ax1.set_title('Dependent Variable y1')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Actual y1')
    #ax1.set_zlabel('Predicted y1')

    # 3D plot for the second dependent variable
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(X_test_unscaled, y_test_unscaled, y_pred_unscaled, color='red', label='Actual vs Predicted y2')
    ax2.set_title('Dependent Variable y2')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Actual y2')
    #ax2.set_zlabel('Predicted y2')

    plt.show()


csv_file_path = '../dataset/ipc_cycles_dataset/ML_model_ipc_cycles_dataset_10 iterations_avg.csv'
data = pd.read_csv(csv_file_path)
print(data)
clean_data_stage1 = clean_data(data,'ipc','cpu energy')
print(clean_data_stage1)
clean_data_stage2 = clean_data(clean_data_stage1,'cycles','dram energy')
print(clean_data_stage2)


X = clean_data_stage2[['cycles','ipc']]
y = clean_data_stage2[['cpu energy','dram energy']]
print(X)
print(y)




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
print("shape")

# Verify the shapes of the splits
#print(X_train_scaled.shape, X_test_scaled.shape, y_train.shape, y_test.shape)

# Define the neural network model
np.random.seed(42)
model = Sequential([
    Dense(50, activation='relu', input_shape=(2,)),
    #Dense(20, activation='relu'),
    Dense(2)  # Output layer
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
print("X_test_unscaled")
print(X_test_unscaled)
print(X_test_unscaled.shape)
print("y test unscaled")
print(y_test_unscaled)
print(y_test_unscaled.shape)
print("y pred unscaled")
print(y_pred_unscaled)
print(y_pred_unscaled.shape)

feature_list = ['cycles', 'ipc']
target_list_test = ['cpu energy (Test)', 'dram energy (Test)']
target_list_pred = ['cpu energy (Predicted)', 'dram energy (Predicted)']

for i in range(len(feature_list)):
    for j in range(len(target_list_test)):
        plot_graph(X_test_unscaled, y_test_unscaled, y_pred_unscaled,i,j, feature_list[i], target_list_test[j], target_list_pred[j])



y_pred_list = []
for i in range(len(y_pred_unscaled)):
    y_pred_list.append(y_pred_unscaled[i][0])

y_test_list = []
for i in range(len(y_test_unscaled)):
    y_test_list.append(y_test_unscaled[i][0])

to_csvfile = {}

#plot_2d_graph(y_test_list,y_pred_list)

to_csvfile["pred"] = y_pred_list
to_csvfile["true"] = y_test_list

df = pd.DataFrame(to_csvfile)
csv_file = '../dataset/ipc_dataset/NN_model_ipc_cpu_cyles_dram_pred_test_huber_loss_compare.csv' # Specify your CSV file name
df.to_csv(csv_file, index=False, mode = 'w')