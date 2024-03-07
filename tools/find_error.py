# Function to calculate the error percentage between two numbers
import statistics

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_2d_graph(X,Y):
    plt.figure(figsize=(10, 10))
    plt.scatter(X, Y, label='graph')

    # Adding title and labels
    plt.title('X vs Y')
    plt.xlabel('X')
    plt.ylabel('Y')

    # Adding a grid for better readability
    plt.grid(True)


    # Showing the plot
    plt.legend()
    plt.show()
def calculate_error_percentage(actual, predicted):
    error_percentage = ((actual - predicted) / actual) * 100
    return abs(error_percentage)  # Taking the absolute value to get a positive percentage

data = pd.read_csv("../dataset/ipc_dataset/NN_model_ipc_cpu_pred_test_compare.csv")
print(data['true'])
print(data['pred'])
plot_2d_graph(data['pred'],data['true'])
"""data['true'] = pd.to_numeric(data['true'], errors='coerce')
data['pred'] = pd.to_numeric(data['pred'], errors='coerce')"""

actual = data['true']
predicted = data['pred']
error = []

for i, j in zip(actual, predicted):
    error_percentage = calculate_error_percentage(i, j)/len(actual)
    error.append(error_percentage)

print(error)
"""# Using pandas to ignore NaN
pd_series = pd.Series(error)
sum_without_nan_pd = pd_series.sum()
print("Sum with pandas ignoring NaN:", sum_without_nan_pd)

# Using numpy to ignore NaN
sum_without_nan_np = np.nansum(error)
print("Sum with numpy ignoring NaN:", sum_without_nan_np)

# If you need to remove NaN values explicitly
clean_list = pd_series.dropna().tolist()
print("List without NaN values:", clean_list)
sum_clean_list = sum(clean_list)
print("Sum of cleaned list:", sum_clean_list)"""


print(np.average(error))
