import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
from scipy import stats

# Path to your CSV file
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import dataset

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

# Creating a DataFrame from the CSV data
data = pd.read_csv("../dataset/ipc_cycles_dataset/combined.csv")
print(data)
clean_data_stage1 = clean_data(data,'ins','cpu energy')
print(clean_data_stage1)
clean_data_stage2 = clean_data(clean_data_stage1,'cycles','dram energy')
print(clean_data_stage2)


X = clean_data_stage2[['ins']]
Y = clean_data_stage2[['cpu energy']]
print(X)
print(Y)

"""#For CPU Model
# Independent variables
X = data['ipc']

# Dependent variables
Y = data['cpu energy']"""


"""#For DRAM Model

independent_vars = ['L1 Cache Misses', 'Conditional branch ins', 'Total Cycles']

# Dependent variables
dependent_var = 'dram energy'"""


"""
# Create three subplots
fig, axs = plt.subplots(1, 1, figsize=(15, 5), sharey=True)

# Iterate through each independent variable and create scatter plots
for i, independent_var in enumerate(independent_vars):
    axs[i].plot(data[independent_var], data[dependent_var])
    axs[i].set_xlabel(independent_var)

# Set common y-axis label
axs[0].set_ylabel(dependent_var)

# Set the overall title for the subplots
fig.suptitle(f"Scatter Plots of Independent Variables vs {dependent_var}")

# Show the plots
plt.show()
"""

""" Create a 3x3 subplot
fig, axs = plt.subplots(1, 3, figsize=(10, 10))

# Loop through each independent variable
for i, ivar in enumerate(independent_vars):
    # Loop through each dependent variable
    for j, dvar in enumerate(dependent_vars):
        # Create scatter plot for each combination of independent and dependent variables
        axs[j, i].scatter(data[ivar], data[dvar])
        axs[j, i].set_xlabel(ivar)
        axs[j, i].set_ylabel(dvar)
        axs[j, i].set_title(f'{ivar} vs {dvar}')

# Adjust layout for better readability
plt.tight_layout()
plt.show()"""

plt.figure(figsize=(10, 10))
plt.scatter(X, Y, label='graph')

# Adding title and labels
plt.title('perf counters vs energy')
plt.xlabel('ipc')
plt.ylabel('dram energy')

# Adding a grid for better readability
plt.grid(True)

# Showing the plot
plt.legend()
plt.show()




