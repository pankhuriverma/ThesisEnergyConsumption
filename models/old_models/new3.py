import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

class NeuralNetwork:
  def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=0.0001):
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.output_dim = output_dim
    self.learning_rate = learning_rate

    # Initialize weights and biases
    self.w1 = np.random.randn(self.input_dim, self.hidden_dim) * 0.01
    self.b1 = np.zeros((1, self.hidden_dim))
    self.w2 = np.random.randn(self.hidden_dim, self.output_dim) * 0.01
    self.b2 = np.zeros((1, self.output_dim))

  def forward(self, X):
    # Forward pass
    z1 = X.dot(self.w1) + self.b1  # (m, hidden_dim)
    a1 = self.relu(z1)  # (m, hidden_dim)
    z2 = a1.dot(self.w2) + self.b2  # (m, output_dim)
    a2 = z2  # (m, output_dim)  # Linear activation for regression
    cache = {"Z1": z1, "A1": a1, "Z2": z2, "A2": a2}
    return a2, cache

  def backward(self, X, y, predicted, cache):
    # Backpropagation
    # Calculate gradients
    a1, a2 = cache["A1"], cache["A2"]
    da2 = -(y - predicted)  # (m, output_dim)
    dz2 = da2  # (m, output_dim)

    dw2 = np.transpose(a1).dot(dz2)  # (hidden_dim, output_dim)
    db2 = np.sum(dz2, axis=0, keepdims=True)  # (1, output_dim)

    da1 = dz2.dot(np.transpose(self.w2)) * self.relu_derivative(a1)  # (m, hidden_dim)
    dz1 = da1  # (m, hidden_dim)

    dw1 = np.transpose(X).dot(dz1)  # (input_dim, hidden_dim)
    db1 = np.sum(dz1, axis=0, keepdims=True)  # (1, hidden_dim)

    # Update weights and biases
    self.w1 -= self.learning_rate * dw1
    self.b1 -= self.learning_rate * db1
    self.w2 -= self.learning_rate * dw2
    self.b2 -= self.learning_rate * db2

  def predict(self, X):
    # Predict output
    return self.forward(X)

  def relu(self, x):
    return np.maximum(0, x)

  def relu_derivative(self, x):
    return x > 0


  def train(self, X, y, epochs=100):
    # Train the model
    for epoch in range(epochs):
      predicted, cache = self.forward(X)
      loss = np.mean(np.abs(y - predicted))  # Mean Absolute Error
      self.backward(X, y, predicted, cache)
      print(f"Epoch: {epoch+1}/{epochs}, Loss: {loss:.4f}")

    return predicted

def clean_data(data):

    mean = data['cpu energy'].mean()
    std_dev = data['cpu energy'].std()
    threshold = 3  # 3 standard deviations
    data['Z_score'] = (data['cpu energy'] - mean) / std_dev
    cleaned_data = data[np.abs(data['Z_score']) <= threshold].drop(columns=['Z_score'])
    cleaned_data[['ipc']] = cleaned_data[['ipc']].round(4)
    cleaned_data[["cpu energy"]] = cleaned_data[["cpu energy"]].round(4)
    return cleaned_data[['ipc']], cleaned_data[["cpu energy"]]

def test_model(model, X_test, y_test):
  predicted, cache = model.predict(X_test)
  loss = np.mean(np.abs(y_test - predicted))
  print(f"Test Loss: {loss:.4f}")

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

# Model definition
model = NeuralNetwork(1, 100, 1)

csv_file_path = '../../dataset/ipc_dataset/ML_model_collected_dataset_ipc_10iterations_avg.csv'  # Change this to the correct path
data = pd.read_csv(csv_file_path)
#X, y = clean_data(data)
X = data[["ipc"]]
y = data[["cpu energy"]]
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_x.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
print("X_train, Xtest, y_train, y_test")
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

# Train the model
y_pred = model.train(X_train, y_train, 100)  # Change epochs as needed

print(y_pred)
plot_graph(X_train.flatten(), y_train.flatten(), y_pred.flatten())
# Testing function (example)
test_model(model, X_test, y_test)