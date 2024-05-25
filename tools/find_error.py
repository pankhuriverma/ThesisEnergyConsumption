# Function to calculate the error percentage between two numbers
import statistics

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_2d_graph(X,Y):
    plt.figure(figsize=(10, 10))
    plt.scatter(X, Y, label='graph')

    # Adding title and labels
    plt.title('DRAM CO2 Emission (Test vs Predicted)')
    plt.xlabel('DRAM CO2 Emission (Test)')
    plt.ylabel('DRAM CO2 Emission (Predicted)')

    # Adding a grid for better readability
    plt.grid(True)


    # Showing the plot
    plt.legend()
    plt.show()
def calculate_error_percentage(actual, predicted):
    error_percentage = ((actual - predicted) / actual) * 100
    return abs(error_percentage)  # Taking the absolute value to get a positive percentage

data = pd.read_csv("../dataset/energy_model_results_dataset/NN_model_dram_energy_ins_cycles_mean_absolute_error.csv")
print(data['pred co2'])
print(data['true co2'])
plot_2d_graph(data['pred co2'],data['true co2'])
"""data['true'] = pd.to_numeric(data['true'], errors='coerce')
data['pred'] = pd.to_numeric(data['pred'], errors='coerce')"""

actual_co2 = data['true co2']
predicted_co2 = data['pred co2']

error_co2=[]
for i, j in zip(actual_co2, predicted_co2):
    error_percentage_co2 = calculate_error_percentage(i, j)/len(actual_co2)
    error_co2.append(error_percentage_co2)

print("CO2 Error",np.average(error_co2))

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



